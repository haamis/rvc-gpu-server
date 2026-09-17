"""Owns the persistent RVC worker subprocess (desktop/GPU side).

Port of the worker-management half of TTS-bot's ``ttsbot/rvc/runner.py``,
minus the remote-HTTP client (which lives thin-side). Same protocol, same
failure taxonomy, same memory hygiene:

- JSON-lines protocol (``ok: false`` = deterministic request error ->
  ``RVCRequestError``, fail fast; EOF/desync/timeout = ``WorkerCrashed`` ->
  kill, respawn, retry once, then one-shot subprocess fallback).
- The worker caches ONE model at a time; a failed ``get_vc`` drops the
  cache server-side (see worker.py) — never loosen this.
- ``MALLOC_ARENA_MAX=2`` + ``MALLOC_TRIM_THRESHOLD_`` bound glibc arena
  fragmentation; ``RVC_WORKER_MAX_RSS_MB`` recycles the worker after the
  response is sent (no lost requests).

No third-party imports (stdlib only) so this module is unit-testable
anywhere; the FastAPI shell in server.py is the only web dependency.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("rvc_server.owner")


class RVCRequestError(Exception):
    """Deterministic per-request failure (worker is healthy; retry won't help)."""


class WorkerCrashed(Exception):
    """Worker process crash, protocol desync, or timeout."""


class RvcWorkerOwner:
    """Spawns and serializes one worker.py subprocess (single-job-at-a-time)."""

    def __init__(
        self,
        worker_script: str | Path,
        rvc_root: str | Path,
        use_worker: bool = True,
    ):
        self.worker_script = Path(worker_script)
        self.rvc_root = Path(rvc_root).resolve()
        self.use_worker = use_worker
        self._worker: asyncio.subprocess.Process | None = None
        self._worker_lock = asyncio.Lock()
        self._req_id = 0
        # Last worker ready-handshake ({"device": ..., "threads": ...});
        # surfaced via /health so a silent CPU fallback is visible.
        self.worker_info: dict = {}
        # Model path of the last successful conversion (/health reporting).
        self.last_model: str | None = None

    async def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        model_path: str,
        index_path: str | None,
        pitch: int = 0,
        index_rate: float = 0.75,
        f0_method: str = "rmvpe",
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        speaker_id: int = 0,
    ) -> str:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._check_paths(model_path, index_path)

        if self.use_worker:
            try:
                out = await self._convert_worker(
                    input_path, output_path, model_path, index_path,
                    pitch, index_rate, f0_method, resample_sr,
                    rms_mix_rate, protect, speaker_id,
                )
                self.last_model = str(model_path)
                return out
            except RVCRequestError:
                # Deterministic failure — a subprocess would fail identically.
                raise
            except Exception as e:
                log.warning("Worker conversion failed, falling back to subprocess: %s", e)

        out = await self._convert_subprocess(
            input_path, output_path, model_path, index_path,
            pitch, index_rate, f0_method, resample_sr,
            rms_mix_rate, protect, speaker_id,
        )
        self.last_model = str(model_path)
        return out

    def _check_paths(self, model_path: str, index_path: str | None) -> None:
        if not self.worker_script.exists():
            raise RuntimeError(f"RVC worker script not found: {self.worker_script}")
        if not Path(model_path).exists():
            raise RuntimeError(f"RVC model not found: {model_path}")
        if index_path and not Path(index_path).exists():
            raise RuntimeError(f"RVC index not found: {index_path}")

    # ---- persistent worker mode ----

    async def _ensure_worker(self) -> asyncio.subprocess.Process:
        if self._worker is not None and self._worker.returncode is None:
            return self._worker

        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(self.rvc_root) + (os.pathsep + existing if existing else "")
        env["RVC_WORKER_ROOT"] = str(self.rvc_root)
        # glibc defaults to 8*ncores malloc arenas and rarely returns freed
        # heap; both inflate long-lived worker RSS by gigabytes.
        env.setdefault("MALLOC_ARENA_MAX", "2")
        env.setdefault("MALLOC_TRIM_THRESHOLD_", str(128 * 1024 * 1024))
        env["RVC_WORKER_MAX_RSS_MB"] = os.environ.get("RVC_WORKER_MAX_RSS_MB", "2500")

        self._worker = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(self.worker_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self.rvc_root),
        )
        line = await asyncio.wait_for(self._worker.stdout.readline(), timeout=300)
        if not line:
            raise RuntimeError("worker died during startup")
        resp = json.loads(line)
        if resp.get("event") != "ready":
            raise RuntimeError(f"unexpected worker handshake: {resp}")
        self.worker_info = {
            "device": resp.get("device"),
            "threads": resp.get("threads"),
        }
        log.info("RVC worker ready (device=%s threads=%s)", resp.get("device"), resp.get("threads"))
        return self._worker

    async def _kill_worker(self) -> None:
        if self._worker is None:
            return
        try:
            self._worker.kill()
        except ProcessLookupError:
            pass
        self._worker = None

    async def _convert_worker(
        self, input_path, output_path, model_path, index_path,
        pitch, index_rate, f0_method, resample_sr, rms_mix_rate, protect, speaker_id,
    ) -> str:
        # One in-flight request at a time keeps the protocol simple and gates
        # the single GPU.
        async with self._worker_lock:
            self._req_id += 1
            req = {
                "id": self._req_id,
                "cmd": "convert",
                "input": str(input_path),
                "output": str(output_path),
                "model": str(model_path),
                "index": index_path or "",
                "pitch": int(pitch),
                "index_rate": float(index_rate),
                "f0_method": f0_method,
                "resample_sr": int(resample_sr),
                "rms_mix_rate": float(rms_mix_rate),
                "protect": float(protect),
                "speaker_id": int(speaker_id),
            }

            for attempt in (1, 2):
                try:
                    proc = await self._ensure_worker()
                    proc.stdin.write((json.dumps(req) + "\n").encode())
                    await proc.stdin.drain()

                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=1800)
                    if not line:
                        raise WorkerCrashed("worker died")
                    resp = json.loads(line)
                    if resp.get("id") != req["id"]:
                        raise WorkerCrashed(f"protocol desync: {resp}")
                    if not resp.get("ok"):
                        # Deterministic per-request failure (bad input/model):
                        # the worker is healthy, so fail fast without killing
                        # it or retrying a doomed request.
                        raise RVCRequestError(f"RVC worker error: {resp.get('error')}")
                    if not output_path.exists():
                        raise RVCRequestError(f"RVC produced no output at {output_path}")
                    return str(output_path)
                except WorkerCrashed as e:
                    await self._kill_worker()
                    if attempt == 2:
                        raise RuntimeError(f"RVC worker crashed: {e}") from e
                    log.warning("RVC worker crashed; respawning and retrying once")
                except asyncio.TimeoutError as e:
                    await self._kill_worker()
                    if attempt == 2:
                        raise RuntimeError("RVC worker timed out") from e
                    log.warning("RVC worker timed out; respawning and retrying once")

        raise RuntimeError("unreachable")

    # ---- subprocess mode (fallback) ----

    async def _convert_subprocess(
        self, input_path, output_path, model_path, index_path,
        pitch, index_rate, f0_method, resample_sr, rms_mix_rate, protect, speaker_id,
    ) -> str:
        infer_script = self.rvc_root / "infer" / "cli.py"
        cmd = [
            sys.executable,
            str(infer_script),
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--speaker-id",
            str(speaker_id),
            "--pitch",
            str(pitch),
            "--f0-method",
            f0_method,
            "--index-rate",
            str(index_rate),
            "--protect",
            str(protect),
        ]

        if index_path:
            cmd.extend(["--index", str(index_path)])

        if resample_sr > 0:
            cmd.extend(["--resample-sr", str(resample_sr)])

        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(self.rvc_root) + (os.pathsep + existing if existing else "")
        # The vendored CLI loads checkpoints with legacy torch.load; silence
        # the FutureWarning without touching the submodule.
        env["PYTHONWARNINGS"] = "ignore::FutureWarning" + (
            "," + env["PYTHONWARNINGS"] if env.get("PYTHONWARNINGS") else ""
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self.rvc_root),
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip()
            raise RuntimeError(f"RVC inference failed (exit {proc.returncode}): {err}")

        if not output_path.exists():
            raise RuntimeError(f"RVC produced no output at {output_path}")

        return str(output_path)

    async def shutdown(self) -> None:
        """Terminate the persistent worker, if running."""
        if self._worker is None:
            return
        try:
            if self._worker.returncode is None:
                self._worker.stdin.write(b'{"id": 0, "cmd": "shutdown"}\n')
                await self._worker.stdin.drain()
                try:
                    await asyncio.wait_for(self._worker.wait(), timeout=10)
                except asyncio.TimeoutError:
                    self._worker.kill()
        except Exception:
            try:
                self._worker.kill()
            except ProcessLookupError:
                pass
        finally:
            self._worker = None
