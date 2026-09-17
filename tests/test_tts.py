"""POST /tts contract tests (server venv).

kokoro_engine.synthesize is stubbed (no weights/GPU); error mapping and
auth are real. Run: pytest tests/ -q from the repo root.
"""

import io
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

import server
from server import PROTOCOL_HEADER, PROTOCOL_VERSION, create_app
import kokoro_engine


def _sine_wav_bytes(sr=24000, secs=1.0):
    buf = io.BytesIO()
    t = np.arange(int(sr * secs)) / sr
    sf.write(buf, 0.5 * np.sin(2 * np.pi * 220 * t), sr, format="WAV")
    return buf.getvalue()


def _client(tmp_path, monkeypatch, behavior="ok", token=""):
    def fake_synth(text, voice, speed, out):
        if behavior == "bad-voice":
            raise RuntimeError(f"Unknown Kokoro voice {voice!r}")
        if behavior == "crash":
            raise RuntimeError("onnx exploded")
        Path(out).write_bytes(_sine_wav_bytes())
        return str(out)

    monkeypatch.setattr(kokoro_engine, "synthesize", fake_synth)
    app = create_app(rvc_root=str(tmp_path), require_token=token, start_worker=False)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


async def _post(client, extra_data=None, extra_headers=None):
    headers = {PROTOCOL_HEADER: PROTOCOL_VERSION}
    headers.update(extra_headers or {})
    data = {"text": "hello world", "voice": "af_heart", "speed": "1.0"}
    data.update(extra_data or {})
    return await client.post("/tts", data=data, headers=headers)


async def test_tts_roundtrip(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = await _post(client)
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 24000 and float(np.sqrt(np.mean(data**2))) > 0.005


async def test_tts_bad_voice_is_400(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, behavior="bad-voice")
    r = await _post(client)
    assert r.status_code == 400


async def test_tts_crash_is_503(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, behavior="crash")
    r = await _post(client)
    assert r.status_code == 503


async def test_tts_auth_and_protocol(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, token="sekret")
    assert (await _post(client)).status_code == 401
    bad = await _post(
        client,
        extra_headers={"Authorization": "Bearer sekret", PROTOCOL_HEADER: "9"},
    )
    assert bad.status_code == 400
    ok = await _post(client, extra_headers={"Authorization": "Bearer sekret"})
    assert ok.status_code == 200
