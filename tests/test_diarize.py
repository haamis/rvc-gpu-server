"""POST /diarize contract tests (server venv).

analyze_media is stubbed (no weights); the thin client's finalize path is
covered thin-side. Run: pytest tests/ -q from the repo root.
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
from ttsbot.media.diarize import DiarizationAnalysis, Segment


def _analysis():
    return DiarizationAnalysis(
        segments=[Segment(0.0, 2.5, 0), Segment(3.0, 5.0, 1)],
        cluster_f0={0: 110.0, 1: None},
        detected=2,
    )


def _client(tmp_path, monkeypatch, behavior="ok", token=""):
    def fake_analyze(path, num_voices, threshold, device="cpu"):
        assert num_voices == 2
        if behavior == "data-error":
            raise ValueError("no speech detected in media")
        if behavior == "crash":
            raise RuntimeError("wav2vec2 exploded")
        return _analysis()

    monkeypatch.setattr(server, "analyze_media", fake_analyze)
    app = create_app(rvc_root=str(tmp_path), require_token=token, start_worker=False)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


def _wav_bytes(sr=16000, secs=1.0):
    buf = io.BytesIO()
    t = np.arange(int(sr * secs)) / sr
    sf.write(buf, 0.5 * np.sin(2 * np.pi * 220 * t), sr, format="WAV")
    return buf.getvalue()


async def _post(client, extra_data=None, extra_headers=None):
    headers = {PROTOCOL_HEADER: PROTOCOL_VERSION}
    headers.update(extra_headers or {})
    data = {"engine": "local", "num_voices": "2", "threshold": "0.35"}
    data.update(extra_data or {})
    return await client.post(
        "/diarize",
        files={"audio": ("in.wav", _wav_bytes(), "audio/wav")},
        data=data,
        headers=headers,
    )


async def test_diarize_roundtrip(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = await _post(client)
    assert r.status_code == 200
    body = r.json()
    assert body["segments"] == [
        {"start": 0.0, "end": 2.5, "cluster": 0},
        {"start": 3.0, "end": 5.0, "cluster": 1},
    ]
    assert body["cluster_f0"] == {"0": 110.0, "1": None}
    assert body["detected"] == 2
    assert body["engine"] == "local"


async def test_diarize_rejects_bad_engine(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = await _post(client, extra_data={"engine": "pyannote"})
    assert r.status_code == 501


async def test_diarize_data_error_is_400(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, behavior="data-error")
    r = await _post(client)
    assert r.status_code == 400


async def test_diarize_crash_is_503(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, behavior="crash")
    r = await _post(client)
    assert r.status_code == 503


async def test_diarize_auth(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, token="sekret")
    assert (await _post(client)).status_code == 401
    ok = await _post(client, extra_headers={"Authorization": "Bearer sekret"})
    assert ok.status_code == 200
