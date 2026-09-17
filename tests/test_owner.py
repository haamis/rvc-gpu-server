"""worker_owner.py contract tests — no torch, no fastapi, no network.

A stub worker script (stdlib only) speaks the JSON-lines protocol; behaviors
are switched via STUB_MODE env: ok | request-error | crash.
"""

import asyncio
import json
import math
import os
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker_owner import RVCRequestError, RvcWorkerOwner

STUB = """
import json, os, sys, wave, math, struct
sys.stdout.write(json.dumps({"event": "ready", "device": "stub", "threads": 1}) + "\\n")
sys.stdout.flush()
mode = os.environ.get("STUB_MODE", "ok")
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("cmd") == "shutdown":
        sys.stdout.write(json.dumps({"id": req.get("id"), "ok": True}) + "\\n")
        sys.stdout.flush()
        break
    if req.get("cmd") != "convert":
        sys.stdout.write(json.dumps({"id": req.get("id"), "ok": False, "error": "bad cmd"}) + "\\n")
        sys.stdout.flush()
        continue
    if mode == "crash":
        os._exit(1)
    if mode == "request-error" or not os.path.exists(req["input"]):
        sys.stdout.write(json.dumps({"id": req["id"], "ok": False, "error": "nope"}) + "\\n")
        sys.stdout.flush()
        continue
    sr, secs, freq = 16000, 1.0, 440.0
    n = int(sr * secs)
    with wave.open(req["output"], "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(
            struct.pack("<h", int(16000 * math.sin(2 * math.pi * freq * i / sr)))
            for i in range(n)
        ))
    sys.stdout.write(json.dumps({"id": req["id"], "ok": True}) + "\\n")
    sys.stdout.flush()
"""


@pytest.fixture
def stub_dir(tmp_path):
    (tmp_path / "stub_worker.py").write_text(STUB)
    (tmp_path / "rvc_root").mkdir()
    return tmp_path


def _owner(stub_dir, **kw):
    env_mode = kw.pop("stub_mode", None)
    if env_mode is not None:
        os.environ["STUB_MODE"] = env_mode
    else:
        os.environ.pop("STUB_MODE", None)
    return RvcWorkerOwner(
        worker_script=stub_dir / "stub_worker.py",
        rvc_root=stub_dir / "rvc_root",
        **kw,
    )


def _wav_in(path):
    sr, secs, freq = 16000, 0.5, 330.0
    n = int(sr * secs)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(
            struct.pack("<h", int(16000 * math.sin(2 * math.pi * freq * i / sr)))
            for i in range(n)
        ))
    return str(path)


def _rms(path):
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5 / 32768


def test_convert_success(stub_dir):
    async def go():
        owner = _owner(stub_dir)
        model = stub_dir / "voice.pth"
        model.write_bytes(b"x")
        src = _wav_in(stub_dir / "in.wav")
        out = await owner.convert(src, stub_dir / "out.wav", str(model), None, pitch=-5)
        assert out == str(stub_dir / "out.wav")
        assert _rms(out) > 0.05
        assert owner.worker_info["device"] == "stub"
        assert owner.last_model == str(model)
        await owner.shutdown()
        assert owner._worker is None

    asyncio.run(go())


def test_request_error_fails_fast_worker_survives(stub_dir):
    async def go():
        owner = _owner(stub_dir)
        model = stub_dir / "voice.pth"
        model.write_bytes(b"x")
        with pytest.raises(RVCRequestError):
            await owner.convert(str(stub_dir / "missing.wav"), stub_dir / "o.wav",
                                str(model), None)
        # Worker still alive: a good request succeeds without respawn.
        src = _wav_in(stub_dir / "in.wav")
        out = await owner.convert(src, stub_dir / "o2.wav", str(model), None)
        assert Path(out).exists()
        await owner.shutdown()

    asyncio.run(go())


def test_stub_mode_request_error(stub_dir):
    async def go():
        owner = _owner(stub_dir, stub_mode="request-error")
        model = stub_dir / "voice.pth"
        model.write_bytes(b"x")
        src = _wav_in(stub_dir / "in.wav")
        with pytest.raises(RVCRequestError):
            await owner.convert(src, stub_dir / "o.wav", str(model), None)
        await owner.shutdown()

    asyncio.run(go())


def test_crash_respawns_retries_then_subprocess_fallback(stub_dir):
    async def go():
        owner = _owner(stub_dir, stub_mode="crash")
        model = stub_dir / "voice.pth"
        model.write_bytes(b"x")
        src = _wav_in(stub_dir / "in.wav")
        # Worker crashes twice (initial + respawned retry) -> owner exhausts
        # the worker path and falls back to the one-shot subprocess, which
        # fails here only because the stub rvc_root has no infer/cli.py.
        # That failure PROVES the fallback was attempted.
        with pytest.raises(RuntimeError, match="RVC inference failed"):
            await owner.convert(src, stub_dir / "o.wav", str(model), None)
        assert owner._worker is None
        await owner.shutdown()

    asyncio.run(go())


def test_missing_model_is_deterministic(stub_dir):
    async def go():
        owner = _owner(stub_dir)
        src = _wav_in(stub_dir / "in.wav")
        with pytest.raises(RuntimeError, match="model not found"):
            await owner.convert(src, stub_dir / "o.wav", str(stub_dir / "nope.pth"), None)
        await owner.shutdown()

    asyncio.run(go())
