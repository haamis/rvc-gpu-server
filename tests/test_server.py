"""server.py endpoint contract tests — needs fastapi/httpx (desktop venv).

Worker spawn disabled; the owner is stubbed. No sockets, no weights.
Run: pytest tests/ -q (from the repo root, with the server venv active).
"""

import io
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import PROTOCOL_HEADER, PROTOCOL_VERSION, create_app
from worker_owner import RVCRequestError, WorkerCrashed

import numpy as np
import soundfile as sf


def _sine_wav(path, sr=16000, secs=1.0, freq=440.0):
    t = np.arange(int(sr * secs)) / sr
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * freq * t), sr)
    return str(path)


def _sine_bytes(sr=16000, secs=1.0, freq=440.0):
    buf = io.BytesIO()
    t = np.arange(int(sr * secs)) / sr
    sf.write(buf, 0.5 * np.sin(2 * np.pi * freq * t), sr, format="WAV")
    return buf.getvalue()


class _FakeProc:
    pid = 1234


class _FakeOwner:
    """Stub for the server-owned RvcWorkerOwner (no subprocess, no weights)."""

    worker_info = {"device": "cuda", "threads": 4}

    def __init__(self, behavior="ok"):
        self.behavior = behavior
        self.seen_kwargs = None
        self.last_model = None

    async def _ensure_worker(self):
        if self.behavior == "dead":
            raise RuntimeError("worker died during startup")
        return _FakeProc()

    async def convert(self, **kwargs):
        self.seen_kwargs = kwargs
        if self.behavior == "request-error":
            raise RVCRequestError("bad audio")
        if self.behavior == "crash":
            raise WorkerCrashed("worker died")
        _sine_wav(kwargs["output_path"])
        self.last_model = kwargs["model_path"]
        return str(kwargs["output_path"])


def _client(tmp_path, behavior="ok", token=""):
    model = tmp_path / "snake.pth"
    model.write_bytes(b"fake-weights")
    app = create_app(rvc_root=str(tmp_path), require_token=token, start_worker=False)
    app.state.owner = _FakeOwner(behavior)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )
    return client, model


async def _post(client, wav_bytes, model, extra_headers=None):
    headers = {PROTOCOL_HEADER: PROTOCOL_VERSION}
    headers.update(extra_headers or {})
    return await client.post(
        "/convert",
        files={"audio": ("in.wav", wav_bytes, "audio/wav")},
        data={"model": str(model), "pitch": "-5", "f0_method": "pm"},
        headers=headers,
    )


def test_check_auth():
    from fastapi import HTTPException

    from server import check_auth

    check_auth(None, "")
    check_auth("Bearer abc", "abc")
    with pytest.raises(HTTPException) as ei:
        check_auth(None, "abc")
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException):
        check_auth("Bearer wrong", "abc")


def test_check_protocol():
    from fastapi import HTTPException

    from server import check_protocol

    check_protocol(PROTOCOL_VERSION)
    with pytest.raises(HTTPException) as ei:
        check_protocol("999")
    assert ei.value.status_code == 400


def test_resolve_model_path(tmp_path):
    from fastapi import HTTPException

    from server import resolve_model_path

    verbatim = tmp_path / "voice.pth"
    verbatim.write_bytes(b"x")
    assert resolve_model_path(str(verbatim), "") == verbatim

    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    (flat_dir / "other.pth").write_bytes(b"x")
    assert resolve_model_path("/elsewhere/other.pth", str(flat_dir)) == flat_dir / "other.pth"

    with pytest.raises(HTTPException) as ei:
        resolve_model_path("/elsewhere/missing.pth", str(flat_dir))
    assert ei.value.status_code == 400


def test_resolve_model_path_recursive_subdir(tmp_path):
    from server import _build_model_index, resolve_model_path

    nested = tmp_path / "voices" / "snake"
    nested.mkdir(parents=True)
    pth = nested / "SSNAKE.pth"
    pth.write_bytes(b"x")
    index = _build_model_index(str(tmp_path / "voices"))
    assert resolve_model_path("/thin/rvc_models/snake/SSNAKE.pth",
                              str(tmp_path / "voices"), index) == pth


async def test_health_ok(tmp_path):
    client, _ = _client(tmp_path)
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["protocol"] == PROTOCOL_VERSION
    assert body["device"] == "cuda"
    assert body["worker_pid"] == 1234


async def test_health_degraded_when_worker_dead(tmp_path):
    client, _ = _client(tmp_path, behavior="dead")
    r = await client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"


async def test_convert_roundtrip(tmp_path):
    client, model = _client(tmp_path)
    r = await _post(client, _sine_bytes(), model)
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 16000 and float(np.sqrt(np.mean(data**2))) > 0.005

    fake = client._transport.app.state.owner
    assert fake.seen_kwargs["model_path"] == str(model)
    assert fake.seen_kwargs["pitch"] == -5
    assert fake.seen_kwargs["f0_method"] == "pm"

    health = (await client.get("/health")).json()
    assert health["loaded_model"] == str(model)


async def test_convert_rejects_bad_protocol(tmp_path):
    client, model = _client(tmp_path)
    r = await _post(client, _sine_bytes(), model, extra_headers={PROTOCOL_HEADER: "999"})
    assert r.status_code == 400


async def test_convert_auth(tmp_path):
    client, model = _client(tmp_path, token="sekret")
    assert (await _post(client, _sine_bytes(), model)).status_code == 401
    ok = await _post(client, _sine_bytes(), model,
                     extra_headers={"Authorization": "Bearer sekret"})
    assert ok.status_code == 200


async def test_convert_missing_model_is_400(tmp_path):
    client, _ = _client(tmp_path)
    r = await _post(client, _sine_bytes(), "/elsewhere/missing.pth")
    assert r.status_code == 400


async def test_convert_maps_request_error_to_400(tmp_path):
    client, model = _client(tmp_path, behavior="request-error")
    r = await _post(client, _sine_bytes(), model)
    assert r.status_code == 400


async def test_convert_maps_crash_to_503(tmp_path):
    client, model = _client(tmp_path, behavior="crash")
    r = await _post(client, _sine_bytes(), model)
    assert r.status_code == 503


async def test_convert_scratch_cleaned(tmp_path, monkeypatch):
    import tempfile

    scratch_root = tmp_path / "tmproot"
    scratch_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_root))
    client, model = _client(tmp_path)
    r = await _post(client, _sine_bytes(), model)
    assert r.status_code == 200
    server_dir = scratch_root / "rvc_server"
    leftovers = list(server_dir.rglob("*")) if server_dir.exists() else []
    assert leftovers == []
