"""Wav slicing/dicing for multi-voice !rvc: load, resample, slice, concat.

All functions are pure numpy/soundfile — no model or network dependencies —
so they are unit-testable in isolation.
"""
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_mono(path: str) -> tuple[np.ndarray, int]:
    """Read any soundfile-decodable file as float32 mono, native rate."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(sr)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample; returns the input unchanged when rates match."""
    if sr_in == sr_out or audio.size == 0:
        return audio
    g = math.gcd(sr_in, sr_out)
    out = resample_poly(audio, sr_out // g, sr_in // g)
    return out.astype(np.float32)


def slice_audio(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Clipped [start_s, end_s) slice; tolerates out-of-range bounds."""
    n = audio.shape[0]
    a = max(0, int(round(start_s * sr)))
    b = min(n, int(round(end_s * sr)))
    return audio[a:b]


def concat_audio(parts: list[np.ndarray], gap_s: float = 0.0, sr: int = 44100) -> np.ndarray:
    """Concatenate parts, optionally separating them with `gap_s` of silence."""
    parts = [p for p in parts if p is not None and p.size]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    if gap_s > 0 and len(parts) > 1:
        gap = np.zeros(int(round(gap_s * sr)), dtype=np.float32)
        joined: list[np.ndarray] = []
        for i, p in enumerate(parts):
            if i:
                joined.append(gap)
            joined.append(p)
        parts = joined
    return np.concatenate(parts).astype(np.float32)


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    sf.write(path, audio, sr, subtype="PCM_16")
