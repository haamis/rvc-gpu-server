"""Kokoro TTS engine (desktop/GPU side).

Thin wrapper over kokoro_onnx with a lazily loaded singleton: kokoro-onnx
picks CUDA automatically when onnxruntime-gpu is installed (ONNX_PROVIDER
overrides). No FastAPI imports — unit-testable anywhere with the deps.
"""

import logging
import os
import threading
from pathlib import Path

import soundfile as sf

log = logging.getLogger("rvc_server.kokoro")

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("KOKORO_MODEL_DIR") or (REPO_ROOT / "models" / "kokoro"))
MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"

_LOCK = threading.Lock()
_KOKORO = None


def model_paths() -> tuple[Path, Path]:
    return MODELS_DIR / MODEL_FILE, MODELS_DIR / VOICES_FILE


def _get():
    global _KOKORO
    if _KOKORO is not None:
        return _KOKORO
    with _LOCK:
        if _KOKORO is not None:
            return _KOKORO
        from kokoro_onnx import Kokoro

        model, voices = model_paths()
        if not model.exists() or not voices.exists():
            raise RuntimeError(f"Kokoro models missing in {MODELS_DIR}")
        _KOKORO = Kokoro(str(model), str(voices))
        return _KOKORO


def available_voices() -> list[str]:
    return _get().get_voices()


def synthesize(text: str, voice: str, speed: float, output_path: str | Path) -> str:
    """Text -> wav file. Raises RuntimeError (deterministic: bad voice/text)."""
    if not text or not text.strip():
        raise ValueError("Kokoro needs non-empty text")
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kokoro = _get()
    if voice not in kokoro.get_voices():
        raise RuntimeError(f"Unknown Kokoro voice {voice!r}")
    samples, sr = kokoro.create(text, voice=voice, speed=float(speed), lang="en-us")
    sf.write(str(output_path), samples, sr)
    if not output_path.exists():
        raise RuntimeError(f"Kokoro produced no output at {output_path}")
    return str(output_path)
