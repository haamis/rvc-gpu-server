"""Separation HTTP contract and PyMSS adapter; no model downloads."""
import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


def wav_bytes():
    out = io.BytesIO()
    sf.write(out, np.zeros(16000), 16000, format="WAV")
    return out.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("error,status", [(None, 200), (ValueError("bad model"), 400),
                                         (FileNotFoundError("weights"), 503),
                                         (RuntimeError("CUDA OOM"), 503)])
async def test_separate_contract(tmp_path, monkeypatch, error, status):
    audio = wav_bytes()

    def separate(data, model):
        assert data == audio
        assert model == "vocals-bs-roformer-368"
        if error:
            raise error
        return {f"{stem}_b64": base64.b64encode(audio).decode()
                for stem in ("vocals", "instrumental")}

    monkeypatch.setattr(server, "_separate_vocals_sync", separate)
    app = server.create_app(tmp_path, require_token="test", start_worker=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        files = {"audio": ("in.wav", audio, "audio/wav")}
        headers = {server.PROTOCOL_HEADER: server.PROTOCOL_VERSION}
        assert (await client.post("/separate", files=files, headers=headers)).status_code == 401
        headers["Authorization"] = "Bearer test"
        response = await client.post("/separate", files=files, headers=headers)
        assert response.status_code == status
        if status == 200:
            for stem in ("vocals", "instrumental"):
                assert base64.b64decode(response.json()[f"{stem}_b64"]) == audio


def test_pymss_adapter(tmp_path, monkeypatch):
    audio = wav_bytes()
    scratch = []
    spec = SimpleNamespace(desired_stem="vocals", secondary_stem="instrumental",
                           desired_suffix="vocals", secondary_suffix="instrumental")

    class Batch:
        def __init__(self, model, fmt, vocals, instrumental):
            assert model is spec and fmt == "wav"
            self.roots = (Path(vocals), Path(instrumental))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def separate_file(self, src):
            scratch.append(Path(src).parent)
            assert Path(src).read_bytes() == audio
            paths = []
            for root, stem in zip(self.roots, ("vocals", "instrumental")):
                root.mkdir()
                path = root / f"input_{stem}.wav"
                path.write_bytes(audio)
                paths.append(str(path))
            return {"outputs": paths}

    monkeypatch.setenv("RVC_WORKER_ROOT", str(tmp_path))
    monkeypatch.delenv("weight_pymss_root", raising=False)
    monkeypatch.setitem(sys.modules, "tools.pymss_webui",
                        SimpleNamespace(MSSTBatchSeparator=Batch, resolve_model=lambda _: spec))
    result = server._separate_vocals_sync(audio, "vocals-bs-roformer-368")
    assert base64.b64decode(result["vocals_b64"]) == audio
    assert base64.b64decode(result["instrumental_b64"]) == audio
    assert not scratch[0].exists()
    with pytest.raises(ValueError, match="invalid separation audio"):
        server._separate_vocals_sync(b"not audio", "vocals-bs-roformer-368")
