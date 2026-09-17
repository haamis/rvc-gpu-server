"""Persistent RVC inference worker.

Reads one JSON request per stdin line and writes one JSON response per line to
stdout. Torch and the RVC models (ContentVec, RMVPE, .pth voices) stay loaded
across requests, eliminating the ~14s of imports + model loading that a fresh
subprocess pays on every conversion.

Requests:
  {"id": 1, "cmd": "ping"}
  {"id": 2, "cmd": "convert", "input": "...", "output": "...", "model": "...",
   "index": "...", "pitch": 0, "index_rate": 0.75, "f0_method": "rmvpe",
   "resample_sr": 0, "rms_mix_rate": 0.25, "protect": 0.33, "speaker_id": 0}
  {"id": 3, "cmd": "shutdown"}

Responses:
  {"event": "ready", ...}                (once, at startup)
  {"id": 2, "ok": true}
  {"id": 2, "ok": false, "error": "..."}
"""
import contextlib
import ctypes
import gc
import io
import json
import logging
import os
import sys
import traceback
from functools import wraps
from pathlib import Path

# Honor a custom rvc_root passed by the runner (defaults to the standard layout)
RVC_ROOT = Path(
    os.environ.get("RVC_WORKER_ROOT")
    or (Path(__file__).resolve().parents[2] / "rvc_infer")
)

_log = logging.getLogger("ttsbot.rvc.worker")


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm", "rb") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return 0.0


def _make_trim_heap():
    try:
        libc = ctypes.CDLL("libc.so.6")

        def _trim_heap() -> None:
            try:
                libc.malloc_trim(0)
            except Exception:
                pass

        return _trim_heap
    except Exception:
        def _trim_heap() -> None:
            pass

    return _trim_heap


_trim_heap = _make_trim_heap()


def main() -> None:
    sys.path.insert(0, str(RVC_ROOT))
    os.chdir(RVC_ROOT)

    import torch

    threads = int(os.environ.get("RVC_WORKER_THREADS") or min(8, os.cpu_count() or 4))
    torch.set_num_threads(max(1, threads))

    # torch.load without weights_only emits a FutureWarning on every model
    # load. RVC checkpoints and rmvpe.pt are plain tensor/primitive payloads,
    # so default to the strict loader; if a checkpoint genuinely needs the
    # legacy unpickler, fall back to it so loading never breaks.
    _orig_torch_load = torch.load

    @wraps(_orig_torch_load)
    def _strict_torch_load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            try:
                return _orig_torch_load(*args, **{**kwargs, "weights_only": True})
            except Exception:
                _log.debug("weights_only load failed; retrying legacy torch.load")
                kwargs.pop("weights_only", None)
                return _orig_torch_load(*args, **kwargs)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _strict_torch_load

    from infer.cli import create_config, write_audio
    from infer.vc.modules import VC

    stdout = sys.stdout

    def send(obj: dict) -> None:
        stdout.write(json.dumps(obj) + "\n")
        stdout.flush()

    config = create_config()
    vc = VC(config)
    # get_vc swaps the shared VC instance's net_g wholesale, so only ONE
    # model is loaded at a time. Track that one, not "ever loaded" — a set
    # would let alternating-voice requests skip reloading and run with the
    # previous request's weights.
    current_model: str | None = None
    max_rss_mb = int(os.environ.get("RVC_WORKER_MAX_RSS_MB") or 0)

    send({"event": "ready", "device": str(config.device), "threads": torch.get_num_threads()})

    status = result = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            send({"id": None, "ok": False, "error": "bad json line"})
            continue

        rid = req.get("id")
        try:
            cmd = req.get("cmd")
            if cmd == "ping":
                send({"id": rid, "ok": True})
                continue
            if cmd == "shutdown":
                send({"id": rid, "ok": True})
                return
            if cmd != "convert":
                send({"id": rid, "ok": False, "error": f"unknown cmd: {cmd!r}"})
                continue

            model_path = req["model"]
            if model_path != current_model:
                os.environ["weight_root"] = str(Path(model_path).parent)
                try:
                    # VC.get_vc prints; keep stdout clean for the protocol
                    with contextlib.redirect_stdout(io.StringIO()):
                        vc.get_vc(Path(model_path).name)
                except Exception:
                    # get_vc mutates the shared VC instance incrementally
                    # (self.cpt is replaced before self.net_g is rebuilt), so
                    # a failed load can leave it half-initialized. Drop the
                    # cache: the model reloads fresh before its next use,
                    # guaranteeing the weights match the request.
                    current_model = None
                    raise
                # load_state_dict copies the weights into net_g, so cpt is a
                # redundant second copy of every tensor (~100-200MB/model)
                vc.cpt = None
                current_model = model_path

            with contextlib.redirect_stdout(io.StringIO()):
                status, result = vc.vc_single(
                    int(req.get("speaker_id", 0)),
                    req["input"],
                    int(req.get("pitch", 0)),
                    req.get("f0_method", "rmvpe"),
                    req.get("index") or "",
                    float(req.get("index_rate", 0.75)),
                    int(req.get("resample_sr", 0)),
                    float(req.get("rms_mix_rate", 0.25)),
                    float(req.get("protect", 0.33)),
                )

            if not result or result[0] is None or result[1] is None:
                raise RuntimeError(f"vc_single failed: {status}")

            write_audio(Path(req["output"]), result[1], result[0], "wav")
            send({"id": rid, "ok": True})
        except Exception:
            send({"id": rid, "ok": False, "error": traceback.format_exc(limit=4)[-400:]})

        # Per-request hygiene: release audio buffers, hand freed heap back to
        # the OS (glibc otherwise ratchets RSS up to the worst-case peak), and
        # recycle the whole process if RSS outgrew the cap. Recycling happens
        # after the response was sent, so no request is ever lost.
        status = result = None
        gc.collect()
        _trim_heap()
        if max_rss_mb and _rss_mb() > max_rss_mb:
            _log.info(
                "worker RSS %.0fMB over %dMB cap; recycling (runner respawns on next request)",
                _rss_mb(), max_rss_mb,
            )
            return


if __name__ == "__main__":
    main()