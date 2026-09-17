"""Chunked-convert math + endpoint tests (server venv).

_plan_chunks/_stitch are pure numpy; the endpoint test uses an identity
owner (copies input->output), so a stitched sine must round-trip exactly.
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
from server import (
    PROTOCOL_HEADER,
    PROTOCOL_VERSION,
    _plan_chunks,
    _stitch,
    create_app,
)


def test_plan_short_input_single_window():
    assert _plan_chunks(100, 200, 20) == [(0, 100)]
    assert _plan_chunks(200, 200, 20) == [(0, 200)]


def test_plan_covers_with_overlap():
    windows = _plan_chunks(1000, 300, 50)
    assert windows[0][0] == 0
    assert windows[-1][1] == 1000
    for (s1, e1), (s2, e2) in zip(windows, windows[1:]):
        assert s2 == e1 - 50  # step = chunk - overlap
        assert e1 - s1 <= 300


def test_plan_degenerate_tail_merged():
    # total=310, chunk=300, overlap=50 -> tail would be 10 samples of pure
    # overlap; merged into one (0, 310) window instead.
    assert _plan_chunks(310, 300, 50) == [(0, 310)]


def test_stitch_single_passthrough():
    a = np.arange(10, dtype=np.float32).reshape(-1, 1)
    out = _stitch([a], 4)
    assert np.array_equal(out, a)


def test_stitch_crossfade_sums_to_one():
    # Constant signal: linear ramps sum to 1, so output == input exactly.
    a = np.full((100, 1), 0.5, dtype=np.float32)
    b = np.full((100, 1), 0.5, dtype=np.float32)
    out = _stitch([a, b], 20)
    assert len(out) == 180
    assert np.allclose(out, 0.5)


def test_stitch_blends_boundary():
    a = np.zeros((50, 1), dtype=np.float32)
    b = np.ones((50, 1), dtype=np.float32)
    out = _stitch([a, b], 10)
    assert len(out) == 90
    assert out[39, 0] == 0.0 and out[50, 0] == 1.0
    mid = out[44, 0]
    assert 0.0 < mid < 1.0  # genuine blend, not a hard cut


class _CopyOwner:
    """Identity 'conversion': output = input (isolates stitch math)."""

    worker_info = {"device": "cuda", "threads": 4}
    last_model = None

    async def _ensure_worker(self):
        class _P:
            pid = 1

        return _P()

    async def convert(self, **kwargs):
        data, sr = sf.read(kwargs["input_path"], dtype="float32", always_2d=True)
        sf.write(kwargs["output_path"], data, sr)
        self.last_model = kwargs["model_path"]
        return str(kwargs["output_path"])


async def test_endpoint_chunks_and_stitches(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CHUNK_SEC", 1.0)
    monkeypatch.setattr(server, "CHUNK_OVERLAP_SEC", 0.25)
    model = tmp_path / "m.pth"
    model.write_bytes(b"x")
    app = create_app(rvc_root=str(tmp_path), start_worker=False)
    app.state.owner = _CopyOwner()
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )
    sr = 16000
    t = np.arange(sr * 3) / sr
    buf = io.BytesIO()
    sf.write(buf, 0.5 * np.sin(2 * np.pi * 440 * t), sr, format="WAV")
    r = await client.post(
        "/convert",
        files={"audio": ("in.wav", buf.getvalue(), "audio/wav")},
        data={"model": str(model)},
        headers={PROTOCOL_HEADER: PROTOCOL_VERSION},
    )
    assert r.status_code == 200
    data, rsr = sf.read(io.BytesIO(r.content))
    assert rsr == sr
    assert abs(len(data) / rsr - 3.0) < 0.05
    assert float(np.sqrt(np.mean(data**2))) > 0.2
