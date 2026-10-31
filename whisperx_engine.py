"""WhisperX alignment engine for word-level timestamps.

Uses faster-whisper (ctranslate2) for transcription + Wav2Vec2 alignment.
Runs on GPU via ctranslate2 CUDA backend. Compatible with pyannote diarization
output for word-level speaker assignment.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("rvc_server.whisperx")

# Lazy import whisperx to avoid circular imports
import importlib
whisperx = importlib.import_module("whisperx")

REPO_ROOT = Path(__file__).resolve().parent

_LOCK = threading.Lock()
_ASR_MODEL = None
_ALIGN_MODEL = None
_ALIGN_METADATA = None
_DEVICE = None
_COMPUTE_TYPE = None


def _resolve_device() -> tuple[str, str]:
    """Return (device, compute_type) for ctranslate2/faster-whisper."""
    global _DEVICE, _COMPUTE_TYPE
    if _DEVICE is not None:
        return _DEVICE, _COMPUTE_TYPE
    device = os.environ.get("RVC_DEVICE", "auto")
    if device == "auto":
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _DEVICE = device
    _COMPUTE_TYPE = "float16" if _DEVICE == "cuda" else "int8"
    log.info("WhisperX device=%s compute_type=%s", _DEVICE, _COMPUTE_TYPE)
    return _DEVICE, _COMPUTE_TYPE


def _get_asr_model():
    """Lazy-load the ASR model (faster-whisper)."""
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return _ASR_MODEL
    with _LOCK:
        if _ASR_MODEL is not None:
            return _ASR_MODEL
        device, compute_type = _resolve_device()
        # Use medium model for balance of accuracy/speed; can be overridden
        model_size = os.environ.get("WHISPERX_MODEL_SIZE", "medium")
        log.info("Loading WhisperX ASR model: %s on %s (%s)", model_size, device, compute_type)
        _ASR_MODEL = whisperx.load_model(model_size, device=device, compute_type=compute_type)
        return _ASR_MODEL


def _get_align_model(language: str = "en"):
    """Lazy-load the alignment model (Wav2Vec2)."""
    global _ALIGN_MODEL, _ALIGN_METADATA
    if _ALIGN_MODEL is not None and _ALIGN_METADATA is not None:
        return _ALIGN_MODEL, _ALIGN_METADATA
    with _LOCK:
        if _ALIGN_MODEL is not None and _ALIGN_METADATA is not None:
            return _ALIGN_MODEL, _ALIGN_METADATA
        device, _ = _resolve_device()
        log.info("Loading WhisperX alignment model for language: %s on %s", language, device)
        _ALIGN_MODEL, _ALIGN_METADATA = whisperx.load_align_model(language_code=language, device=device)
        return _ALIGN_MODEL, _ALIGN_METADATA


def transcribe_and_align(
    audio_path: str,
    language: str = "en",
) -> dict:
    """Full WhisperX pipeline: transcribe -> align.

    Returns dict with segments and words. Speaker assignment is done
    separately via assign_speakers_to_words() using pyannote output.
    """
    device, _ = _resolve_device()

    # 1. Load audio
    audio = whisperx.load_audio(audio_path)

    # 2. Transcribe with faster-whisper
    asr_model = _get_asr_model()
    result = asr_model.transcribe(audio, language=language)

    # 3. Align words with Wav2Vec2
    align_model, align_metadata = _get_align_model(language)
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    return result


def assign_speakers_to_words(
    whisperx_result: dict,
    diarization_segments: list[dict],
) -> dict:
    """Map pyannote diarization segments onto WhisperX word timestamps.

    whisperx_result: output from transcribe_and_align (has 'segments' with 'words')
    diarization_segments: list of {'start': float, 'end': float, 'cluster': int} from pyannote

    Mutates and returns whisperx_result with speaker labels on words/segments.
    """
    # Build a timeline of speaker assignments
    # For each word, find which diarization segment it falls in
    words = []
    for seg_idx, seg in enumerate(whisperx_result.get("segments", [])):
        for w in seg.get("words", []):
            words.append({
                "start": w["start"],
                "end": w["end"],
                "word": w["word"],
                "segment_idx": seg_idx,
            })

    # Assign speaker to each word by overlap
    for w in words:
        w_mid = (w["start"] + w["end"]) / 2
        assigned = None
        for ds in diarization_segments:
            if ds["start"] <= w_mid <= ds["end"]:
                assigned = ds["cluster"]
                break
        w["speaker"] = assigned

    # Propagate speaker to segments (majority vote over words)
    for seg_idx, seg in enumerate(whisperx_result.get("segments", [])):
        seg_words = [w for w in words if w["segment_idx"] == seg_idx]
        if seg_words:
            speakers = [w["speaker"] for w in seg_words if w["speaker"] is not None]
            if speakers:
                seg["speaker"] = max(set(speakers), key=speakers.count)

    return whisperx_result