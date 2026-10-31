"""Chatterbox-Turbo TTS engine (desktop/GPU side).

Thin wrapper over ChatterboxTurboTTS with a lazily loaded singleton.
Supports both Turbo (350M) and Nano (110M) models via the `nano` flag.
Also provides Chatterbox-VC for zero-shot voice conversion.
No FastAPI imports — unit-testable anywhere with the deps.
"""

import logging
import os
import threading
from pathlib import Path

import soundfile as sf
import torch

log = logging.getLogger("rvc_server.chatterbox")

REPO_ROOT = Path(__file__).resolve().parent
# Model cache dir on the GPU server; defaults to HF cache (~/.cache/huggingface)
MODEL_DIR = Path(os.environ.get("CHATTERBOX_MODEL_DIR") or (REPO_ROOT / "models" / "chatterbox"))

_LOCK = threading.Lock()
_TURBO = None
_NANO = None
_VC = None


def _get_turbo():
    global _TURBO
    if _TURBO is not None:
        return _TURBO
    with _LOCK:
        if _TURBO is not None:
            return _TURBO
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        device = os.environ.get("RVC_DEVICE", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading Chatterbox-Turbo on %s", device)
        _TURBO = ChatterboxTurboTTS.from_pretrained(device=device, nano=False)
        return _TURBO


def _get_nano():
    global _NANO
    if _NANO is not None:
        return _NANO
    with _LOCK:
        if _NANO is not None:
            return _NANO
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        device = os.environ.get("RVC_DEVICE", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading Chatterbox-Nano on %s", device)
        _NANO = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
        return _NANO


def _get_vc():
    global _VC
    if _VC is not None:
        return _VC
    with _LOCK:
        if _VC is not None:
            return _VC
        from chatterbox.vc import ChatterboxVC

        device = os.environ.get("RVC_DEVICE", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading Chatterbox-VC on %s", device)
        _VC = ChatterboxVC.from_pretrained(device=device)
        return _VC


def synthesize(
    text: str,
    audio_prompt_path: str | Path | None,
    output_path: str | Path,
    model: str = "turbo",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> str:
    """Text -> wav file. Raises ValueError (deterministic: bad input), RuntimeError otherwise."""
    if not text or not text.strip():
        raise ValueError("Chatterbox needs non-empty text")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model == "nano":
        chatterbox = _get_nano()
    else:
        chatterbox = _get_turbo()

    if audio_prompt_path:
        audio_prompt_path = Path(audio_prompt_path)
        if not audio_prompt_path.exists():
            raise ValueError(f"Reference audio not found: {audio_prompt_path}")

    wav = chatterbox.generate(
        text,
        audio_prompt_path=str(audio_prompt_path) if audio_prompt_path else None,
        exaggeration=float(exaggeration),
        cfg_weight=float(cfg_weight),
    )

    sf.write(str(output_path), wav.squeeze(0).cpu().numpy(), chatterbox.sr)
    if not output_path.exists():
        raise RuntimeError(f"Chatterbox produced no output at {output_path}")
    return str(output_path)


def convert_vc(
    audio_path: str | Path,
    target_voice_path: str | Path,
    output_path: str | Path,
) -> str:
    """Voice conversion: source audio + target voice reference -> converted wav."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_path = Path(audio_path)
    target_voice_path = Path(target_voice_path)
    if not audio_path.exists():
        raise ValueError(f"Source audio not found: {audio_path}")
    if not target_voice_path.exists():
        raise ValueError(f"Target voice reference not found: {target_voice_path}")

    vc = _get_vc()
    wav = vc.generate(audio=str(audio_path), target_voice_path=str(target_voice_path))

    sf.write(str(output_path), wav.squeeze(0).cpu().numpy(), vc.sr)
    if not output_path.exists():
        raise RuntimeError(f"Chatterbox-VC produced no output at {output_path}")
    return str(output_path)