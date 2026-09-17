"""RVC GPU worker server — runs on the desktop machine (Phase 1).

The Discord-facing bot (thin client) uploads source audio via
``POST /convert`` and gets the converted wav back; health/device info via
``GET /health``. See TTS-bot's GPU_UPGRADE_PLAN.md ("Two-machine split").

This repo is self-contained: the FastAPI shell below owns the persistent
worker subprocess (worker.py) through worker_owner.py — no imports from the
TTS-bot repo. The HTTP contract it speaks is versioned; the thin client's
``RvcRunner`` sends ``X-RVC-Protocol: 1`` and this server rejects anything
else with 400. Bump PROTOCOL_VERSION only on breaking changes, and only in
lockstep with the thin client.

Model files: the client sends the model/index paths verbatim (absolute
paths from its voices.yaml). Each path is resolved verbatim first; if
missing and ``RVC_MODEL_ROOT`` is set, ``RVC_MODEL_ROOT/<basename>`` is
tried. So either mirror the thin client's absolute layout or drop all
``.pth``/``.index`` files into one flat ``RVC_MODEL_ROOT`` dir.
Unresolvable paths are a deterministic client error (HTTP 400).

Failure taxonomy (mirrors the JSON-lines worker, different transport):
- 400/401/422 -> deterministic request error, client fails fast.
- 500/503/504, timeouts, connection refused -> transient, client falls
  back to its local CPU worker ("slow path").

Run: ``python server.py`` (env: RVC_GPU_SERVER_HOST/PORT/TOKEN,
RVC_WORKER_ROOT, RVC_WORKER_MAX_RSS_MB, RVC_MODEL_ROOT, RVC_GPU_SERVER_JOB_TIMEOUT).
"""

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from worker_owner import RVCRequestError, RvcWorkerOwner
from ttsbot.media.diarize import analyze_media
import kokoro_engine

log = logging.getLogger("rvc_server")

REPO_ROOT = Path(__file__).resolve().parent

# Must stay in lockstep with the thin client's RvcRunner. Bumped only on
# breaking changes; mismatched clients get 400 (deterministic).
PROTOCOL_VERSION = "1"
PROTOCOL_HEADER = "X-RVC-Protocol"

JOB_TIMEOUT = float(os.getenv("RVC_GPU_SERVER_JOB_TIMEOUT", "1800"))
# Evaluated where the models run (desktop side): "auto" = cuda if available.
RVC_DEVICE = os.getenv("RVC_DEVICE", "auto")
# Long inputs OOM the 8GB card in a single pass (activations scale with
# length), so inputs over CHUNK_SEC are converted in overlapping windows and
# crossfade-stitched. Read dynamically (tests monkeypatch the globals).
CHUNK_SEC = float(os.getenv("RVC_SERVER_CHUNK_SEC", "30"))
CHUNK_OVERLAP_SEC = float(os.getenv("RVC_SERVER_CHUNK_OVERLAP_SEC", "2"))


def check_auth(authorization: str | None, require_token: str) -> None:
    """Raise 401 unless the Bearer token matches (no-op when unset)."""
    if not require_token:
        return
    if authorization != f"Bearer {require_token}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


def check_protocol(protocol: str | None) -> None:
    """Raise 400 when the client's protocol version doesn't match."""
    if protocol != PROTOCOL_VERSION:
        raise HTTPException(
            status_code=400,
            detail=f"protocol mismatch: server={PROTOCOL_VERSION} client={protocol!r}",
        )


def _build_model_index(model_root: str) -> dict[str, list[Path]]:
    """Map basename -> full paths for all .pth/.index files under model_root."""
    index: dict[str, list[Path]] = {}
    for pattern in ("*.pth", "*.index"):
        for p in sorted(Path(model_root).rglob(pattern)):
            index.setdefault(p.name, []).append(p)
    return index


def resolve_model_path(
    requested: str, model_root: str, index: dict[str, list[Path]] | None = None
) -> Path:
    """Resolve a client-sent model/index path against this machine.

    Verbatim first (identical layouts need no config); then basename lookup
    under RVC_MODEL_ROOT (recursive, so the desktop can keep the same
    per-voice subdirs as the thin client). Raises HTTPException(400) when
    unresolvable — a deterministic client error.
    """
    if requested:
        verbatim = Path(requested)
        if verbatim.exists():
            return verbatim
        if model_root:
            flat = Path(model_root) / verbatim.name
            if flat.exists():
                return flat
            if index is None:
                index = _build_model_index(model_root)
            hits = index.get(verbatim.name, [])
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"ambiguous model name {verbatim.name!r} on server: {hits}",
                )
    raise HTTPException(status_code=400, detail=f"model file not found on server: {requested!r}")


def _plan_chunks(total: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """Overlapping [start, end) sample windows covering [0, total).

    Step is chunk-overlap; the last window is clamped to total (so it may be
    shorter). A single window covers short inputs (no stitching).
    """
    if total <= chunk or chunk <= 0:
        return [(0, total)]
    step = max(1, chunk - overlap)
    windows = [(s, min(s + chunk, total)) for s in range(0, total, step)]
    if windows[-1][1] - windows[-1][0] <= overlap and len(windows) > 1:
        # Degenerate tail (all overlap, no new audio): extend the previous
        # window to EOF instead.
        windows[-2] = (windows[-2][0], total)
        windows.pop()
    return windows


def _stitch(chunks: list[np.ndarray], overlap: int) -> np.ndarray:
    """Concatenate chunk outputs with a linear crossfade over `overlap` samples.

    Defensive about RVC output lengths (duration preservation is approximate):
    shortfalls are zero-padded, overruns trimmed.
    """
    if len(chunks) == 1:
        return chunks[0]
    out = chunks[0].copy()
    for nxt in chunks[1:]:
        o = min(overlap, len(out), len(nxt))
        if o > 0:
            # Column-shaped ramp: a flat (o,) would broadcast (o, C) audio
            # into (o, o) — a real bug that shipped in the first version.
            ramp = np.linspace(0.0, 1.0, o).reshape((o,) + (1,) * (out.ndim - 1))
            tail = out[-o:] * (1.0 - ramp) + nxt[:o] * ramp
            out = np.concatenate([out[:-o], tail, nxt[o:]])
        else:
            out = np.concatenate([out, nxt])
    return out


async def _convert_chunked(owner, src: Path, dst: Path, scratch: Path, **params) -> None:
    """Convert, splitting inputs over CHUNK_SEC into overlap-stitched windows."""
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    total = len(audio)
    chunk = int(CHUNK_SEC * sr)
    overlap = int(CHUNK_OVERLAP_SEC * sr)
    windows = _plan_chunks(total, chunk, overlap)
    if len(windows) == 1:
        await owner.convert(input_path=src, output_path=dst, **params)
        return
    log.info("chunked convert: %.1fs -> %d windows", total / sr, len(windows))
    outs: list[np.ndarray] = []
    out_sr = sr
    for i, (s, e) in enumerate(windows):
        c_in = scratch / f"chunk_{i:03d}_in.wav"
        c_out = scratch / f"chunk_{i:03d}_out.wav"
        sf.write(str(c_in), audio[s:e], sr)
        await owner.convert(input_path=c_in, output_path=c_out, **params)
        data, out_sr = sf.read(str(c_out), dtype="float32", always_2d=True)
        outs.append(data)
    o = int(CHUNK_OVERLAP_SEC * out_sr)
    stitched = _stitch(outs, o)
    # Mono-ize only if every chunk came back mono; otherwise keep channels.
    sf.write(str(dst), stitched, out_sr)


def create_app(
    rvc_root: str | Path,
    worker_script: str | Path | None = None,
    require_token: str = "",
    model_root: str = "",
    start_worker: bool = True,
) -> FastAPI:
    """Build the FastAPI app. ``start_worker=False`` skips the eager worker
    spawn (tests); the first /convert or /health then starts it lazily."""

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        if start_worker:
            await _ensure_ready()
            log.info("GPU worker server started (worker=%s)", app.state.ready_info)
        yield

    app = FastAPI(title="RVC GPU worker server", lifespan=_lifespan)
    app.state.owner = RvcWorkerOwner(
        worker_script=worker_script or (REPO_ROOT / "worker.py"),
        rvc_root=str(rvc_root),
        use_worker=True,
    )
    app.state.require_token = require_token
    app.state.model_root = model_root
    app.state.model_index: dict[str, list[Path]] | None = None
    app.state.ready_info: dict | None = None

    async def _ensure_ready() -> dict | None:
        if app.state.ready_info is not None:
            return app.state.ready_info
        try:
            proc = await app.state.owner._ensure_worker()
            app.state.ready_info = {
                "worker_pid": proc.pid,
                # Surfaced in /health so a silent CPU fallback on a CUDA box
                # is visible (rvc_infer degrades defensively); log WARNING.
                "device": app.state.owner.worker_info.get("device"),
                "threads": app.state.owner.worker_info.get("threads"),
            }
            if not str(app.state.ready_info["device"]).startswith("cuda"):
                log.warning(
                    "RVC worker is on device=%s (expected cuda on the GPU box)",
                    app.state.ready_info["device"],
                )
        except Exception as e:
            log.warning("worker not ready: %s", e)
            app.state.ready_info = None
        return app.state.ready_info

    @app.get("/health")
    async def health() -> JSONResponse:
        """Device/queue introspection — the remote ready-handshake line."""
        info = await _ensure_ready()
        if info is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "protocol": PROTOCOL_VERSION},
            )
        return JSONResponse(
            content={
                "status": "ok",
                "protocol": PROTOCOL_VERSION,
                "worker_pid": info.get("worker_pid"),
                "device": info.get("device"),
                "threads": info.get("threads"),
                "loaded_model": app.state.owner.last_model,
            }
        )

    @app.post("/convert")
    async def convert(
        audio: UploadFile = File(...),
        model: str = Form(...),
        index: str = Form(""),
        pitch: int = Form(0),
        index_rate: float = Form(0.75),
        f0_method: str = Form("rmvpe"),
        resample_sr: int = Form(0),
        rms_mix_rate: float = Form(0.25),
        protect: float = Form(0.33),
        speaker_id: int = Form(0),
        keep_files: bool = Form(False),
        authorization: str | None = Header(None),
        protocol: str | None = Header(None, alias=PROTOCOL_HEADER),
    ) -> Response:
        check_protocol(protocol)
        check_auth(authorization, app.state.require_token)

        if app.state.model_index is None and app.state.model_root:
            app.state.model_index = _build_model_index(app.state.model_root)
        model_path = resolve_model_path(model, app.state.model_root, app.state.model_index)
        index_path = (
            resolve_model_path(index, app.state.model_root, app.state.model_index)
            if index else None
        )

        scratch = Path(tempfile.gettempdir()) / "rvc_server" / uuid.uuid4().hex[:12]
        scratch.mkdir(parents=True, exist_ok=True)
        src = scratch / f"input{Path(audio.filename or 'in.wav').suffix or '.wav'}"
        dst = scratch / "converted.wav"
        try:
            src.write_bytes(await audio.read())
            try:
                await asyncio.wait_for(
                    _convert_chunked(
                        app.state.owner,
                        src,
                        dst,
                        scratch,
                        model_path=str(model_path),
                        index_path=str(index_path) if index_path else None,
                        pitch=int(pitch),
                        index_rate=float(index_rate),
                        f0_method=f0_method,
                        resample_sr=int(resample_sr),
                        rms_mix_rate=float(rms_mix_rate),
                        protect=float(protect),
                        speaker_id=int(speaker_id),
                    ),
                    timeout=JOB_TIMEOUT,
                )
            except RVCRequestError as e:
                # Deterministic (bad audio/model): worker is healthy.
                # EXCEPT VRAM exhaustion: that is load/fragmentation
                # dependent, so a retry (locally, or later) can succeed —
                # report it transient so the client degrades to its local
                # worker instead of hard-failing the command.
                if "out of memory" in str(e).lower():
                    raise HTTPException(status_code=503, detail=str(e))
                raise HTTPException(status_code=400, detail=str(e))
            except asyncio.TimeoutError as e:
                raise HTTPException(status_code=504, detail=f"conversion timed out: {e}")
            except Exception as e:
                # Worker crash / desync / unexpected: transient, the client
                # falls back to its local CPU worker.
                raise HTTPException(status_code=503, detail=f"conversion failed: {e}")
            # Buffer the wav so scratch cleanup is deterministic (FileResponse
            # would need the file to outlive the handler).
            return Response(content=dst.read_bytes(), media_type="audio/wav")
        finally:
            if not keep_files:
                shutil.rmtree(scratch, ignore_errors=True)

    @app.post("/diarize")
    async def diarize(
        audio: UploadFile = File(...),
        engine: str = Form("local"),
        num_voices: int = Form(2),
        threshold: float = Form(0.35),
        authorization: str | None = Header(None),
        protocol: str | None = Header(None, alias=PROTOCOL_HEADER),
    ) -> JSONResponse:
        """Speaker analysis for multi-voice !rvc: segments + per-cluster f0.

        Voice assignment (rank matching against the thin client's pitch
        profiles) and gap absorption stay thin-side — see finalize_result.
        Only the local wav2vec2+RMVPE engine runs here for now; anything
        else is 501 (transient: the client falls back to its local engines).
        Data errors (no speech) are 400 (deterministic: no local retry).
        """
        check_protocol(protocol)
        check_auth(authorization, app.state.require_token)
        if engine != "local":
            raise HTTPException(
                status_code=501, detail=f"diarize engine {engine!r} not available on this server"
            )

        scratch = Path(tempfile.gettempdir()) / "rvc_server" / uuid.uuid4().hex[:12]
        scratch.mkdir(parents=True, exist_ok=True)
        src = scratch / f"input{Path(audio.filename or 'in.wav').suffix or '.wav'}"
        try:
            src.write_bytes(await audio.read())
            try:
                analysis = await asyncio.wait_for(
                    asyncio.to_thread(
                        analyze_media,
                        str(src), int(num_voices), float(threshold),
                        RVC_DEVICE,
                    ),
                    timeout=JOB_TIMEOUT,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except asyncio.TimeoutError as e:
                raise HTTPException(status_code=504, detail=f"diarization timed out: {e}")
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"diarization failed: {e}")
            return JSONResponse(
                content={
                    "segments": [
                        {"start": s.start, "end": s.end, "cluster": s.cluster}
                        for s in analysis.segments
                    ],
                    "cluster_f0": {str(c): f for c, f in analysis.cluster_f0.items()},
                    "detected": analysis.detected,
                    "engine": "local",
                }
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    @app.post("/tts")
    async def tts(
        text: str = Form(...),
        voice: str = Form(...),
        speed: float = Form(1.0),
        authorization: str | None = Header(None),
        protocol: str | None = Header(None, alias=PROTOCOL_HEADER),
    ) -> Response:
        """Kokoro TTS on the GPU: text + voice + speed -> wav.

        The thin client's Kokoro-first tier when TTS_PROVIDER=auto: the
        donor prosody for the RVC chain (identity comes from RVC). Speed is
        engine-native. Deterministic errors (bad voice/text) are 400;
        anything else is 503 (client falls through its local tiers).
        """
        check_protocol(protocol)
        check_auth(authorization, app.state.require_token)

        scratch = Path(tempfile.gettempdir()) / "rvc_server" / uuid.uuid4().hex[:12]
        scratch.mkdir(parents=True, exist_ok=True)
        dst = scratch / "tts.wav"
        try:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        kokoro_engine.synthesize,
                        text, voice, float(speed), dst,
                    ),
                    timeout=JOB_TIMEOUT,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except RuntimeError as e:
                # Unknown voice = client config error (fail fast); anything
                # else (missing models, onnx failure) is server-side trouble.
                if "Unknown Kokoro voice" in str(e):
                    raise HTTPException(status_code=400, detail=str(e))
                raise HTTPException(status_code=503, detail=str(e))
            except asyncio.TimeoutError as e:
                raise HTTPException(status_code=504, detail=f"TTS timed out: {e}")
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"TTS failed: {e}")
            return Response(content=dst.read_bytes(), media_type="audio/wav")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    app = create_app(
        rvc_root=os.environ.get("RVC_WORKER_ROOT") or str(REPO_ROOT / "rvc_infer"),
        require_token=os.environ.get("RVC_GPU_SERVER_TOKEN", ""),
        model_root=os.environ.get("RVC_MODEL_ROOT", ""),
    )
    uvicorn.run(
        app,
        host=os.environ.get("RVC_GPU_SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("RVC_GPU_SERVER_PORT", "8001")),
    )


if __name__ == "__main__":
    main()
