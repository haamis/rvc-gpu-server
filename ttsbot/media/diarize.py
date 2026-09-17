"""Speaker diarization for multi-voice !rvc.

Pipeline: energy VAD -> wav2vec2 embeddings over voiced windows ->
agglomerative clustering (cosine; speaker count from the requested voices,
collapsible via RVC_DIARIZE_THRESHOLD) -> segment merging -> per-cluster
median f0 -> pitch-ranked voice assignment.

Model-dependent steps (wav2vec2 embedding, RMVPE f0) are injectable callables
so every decision step is unit-testable without weights or network.

The wav2vec2 model is loaded lazily, only when a multi-voice !rvc actually
runs, and released after each media (the bot process should not hold
~400MB of weights between commands).
"""
import dataclasses
import logging
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from ttsbot.media.audio import (
    concat_audio,
    load_mono,
    resample,
    slice_audio,
    write_wav,
)

log = logging.getLogger("ttsbot.media.diarize")

SAMPLE_RATE = 16000          # wav2vec2 + rmvpe working rate
WINDOW_S = 1.5               # embedding window length
HOP_S = 2.0                  # window stride (voiced regions only)
MIN_SEGMENT_S = 0.4          # shorter speech runs stay original audio
GAP_ABSORB_DB = 45.0         # gap RMS within this of the media peak = audible
MIN_SPEAKER_S = 1.5          # clusters shorter than this are noise, not speakers
MIN_SPEAKER_FRAC = 0.02      # ...or shorter than this fraction of the media
PITCH_WEIGHT = 0.3           # pitch term in the combined clustering distance
PITCH_CAP = 1.6              # max log2 pitch distance contribution (~3 octaves)
F0_ANALYSIS_S = 12.0         # max voiced audio analyzed per cluster
F0_MIN_HZ = 70.0
F0_MAX_HZ = 500.0
F0_MIN_FRAMES = 5            # fewer usable frames -> no trustworthy pitch
RMVPE_HOP = 160              # samples at 16kHz -> 10ms frames

# wav2vec2 final-layer features lean toward ASR content; mid-upper layers
# carry more speaker identity. Index -1 = last of 12 transformer layers;
# recalibrate here if clustering quality disappoints.
_FEATURE_LAYER = -1

# Two multi-voice commands in different channels can run concurrently (the
# channel lock only serializes one channel). This lock guards every
# torch-model init/use in this module (wav2vec2 embed step, RMVPE singleton)
# so two wav2vec2 copies (~360MB each) can never be resident at once.
# RLock: reentrant, so a nested model call can't self-deadlock.
_MODEL_LOCK = threading.RLock()

_RMVPE_MODEL = None
_RMVPE_DEVICE = None


def _resolve_device(device: str) -> str:
    """Map "auto" onto cuda when available (server-side), else the literal."""
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def _get_rmvpe(device: str = "cpu"):
    """Lazy-load the noise-robust RMVPE f0 model from the RVC submodule.

    Reuses the exact model the RVC worker already ships (assets/rmvpe),
    imported the same way the worker does it: namespace packages via
    sys.path, no cwd change. Cached per device (thin client runs cpu, the
    GPU server runs cuda — never both in one process, but safe if so).
    """
    global _RMVPE_MODEL, _RMVPE_DEVICE
    if _RMVPE_MODEL is not None and _RMVPE_DEVICE == device:
        return _RMVPE_MODEL
    with _MODEL_LOCK:
        if _RMVPE_MODEL is not None and _RMVPE_DEVICE == device:
            return _RMVPE_MODEL
        root = Path(
            os.environ.get("RVC_WORKER_ROOT")
            or (Path(__file__).resolve().parents[2] / "rvc_infer")
        )
        model_path = root / "assets" / "rmvpe" / "rmvpe.pt"
        if not root.exists() or not model_path.exists():
            raise RuntimeError(f"rmvpe model not found at {model_path}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from infer.rmvpe import RMVPE

        is_half = str(device).startswith("cuda")
        _RMVPE_MODEL = RMVPE(str(model_path), is_half=is_half, device=device)
        _RMVPE_DEVICE = device
        return _RMVPE_MODEL


def measure_f0(audio: np.ndarray, sr: int, device: str = "cpu") -> float | None:
    """Median f0 (Hz) of voiced frames via RMVPE, or None if untrustworthy.

    RMVPE replaces pyin as the primary estimator: pyin drowns under music
    beds (most frames unvoiced or floor-pinned), RMVPE is trained for
    exactly that and returns 0 for unvoiced frames instead of guessing.
    Falls back to pyin if the RVC submodule/model is unavailable.
    """
    if audio.size < sr // 2:  # < 0.5s: too short to trust
        return None
    if sr != SAMPLE_RATE:
        audio = resample(audio, sr, SAMPLE_RATE)
    try:
        model = _get_rmvpe(device)
    except Exception as e:
        log.warning("RMVPE unavailable (%s); using pyin fallback", e)
        return _measure_f0_pyin(audio, SAMPLE_RATE)
    f0 = model.infer_from_audio(audio.astype(np.float32), thred=0.03)
    n = int(np.ceil(audio.size / RMVPE_HOP))
    vals = np.asarray(f0[:n], dtype=np.float64)
    vals = vals[(vals >= F0_MIN_HZ) & (vals <= F0_MAX_HZ)]
    if vals.size < F0_MIN_FRAMES:
        return None
    return float(np.median(vals))


def _measure_f0_pyin(audio: np.ndarray, sr: int) -> float | None:
    """pyin fallback: median f0 of voiced frames, None if untrustworthy.

    Frames pinning at the fmin floor are discarded: pyin parks unresolvable
    periodicities (music beds, noise) exactly there, and a median of floor
    values is meaningless.
    """
    if audio.size < sr // 2:
        return None
    f0, voiced, _ = librosa.pyin(
        audio, fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, sr=sr, frame_length=1024, hop_length=320
    )
    vals = f0[voiced & np.isfinite(f0)]
    vals = vals[vals > F0_MIN_HZ * 1.05]
    if vals.size < F0_MIN_FRAMES:
        return None
    return float(np.median(vals))


@dataclass
class Segment:
    start: float
    end: float
    cluster: int


@dataclass
class DiarizationResult:
    segments: list[Segment]                 # timeline order
    voice_of_cluster: dict[int, str]
    cluster_f0: dict[int, float | None]
    detected: int                           # distinct clusters found


# --- VAD ---------------------------------------------------------------------


def voice_mask(audio: np.ndarray, sr: int, frame_s: float = 0.03) -> np.ndarray:
    """Per-frame boolean voiced mask: RMS within 40dB of the peak."""
    frame = max(1, int(sr * frame_s))
    n_frames = audio.size // frame
    if n_frames == 0:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt(np.mean(audio[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    db = 20.0 * np.log10(rms + 1e-10)
    return db > (db.max() - 40.0)


def voiced_windows(mask: np.ndarray, duration_s: float, frame_s: float = 0.03,
                   window_s: float = WINDOW_S, hop_s: float = HOP_S) -> list[tuple[float, float]]:
    """[start, end) windows (sorted, non-overlapping) whose frames are >=50% voiced."""
    if mask.size == 0:
        return []
    spans = []
    start = None
    for i, voiced in enumerate(mask):
        if voiced and start is None:
            start = i
        elif not voiced and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, mask.size))

    windows: list[tuple[float, float]] = []
    for fs, fe in spans:
        t0, t1 = fs * frame_s, fe * frame_s
        t = t0
        while t < t1 - 0.5 * frame_s:
            end = min(t + window_s, t1, duration_s)
            if end - t >= 0.5 * window_s:
                windows.append((t, end))
            t += hop_s
    return windows


# --- clustering ----------------------------------------------------------------


def combined_distances(embeddings: np.ndarray, f0s: list[float | None]) -> np.ndarray:
    """Cosine distance + per-window pitch distance.

    The pitch term (PITCH_WEIGHT * clipped |log2 f0 ratio|) separates
    speakers whose voices differ in pitch even when the wav2vec2 embeddings
    interleave (e.g. male+female); windows without a trustworthy f0 fall
    back to embedding distance only.
    """
    norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dist = 1.0 - norms @ norms.T
    if f0s is not None:
        n = len(f0s)
        for i in range(n):
            if not f0s[i]:
                continue
            for j in range(i + 1, n):
                if not f0s[j]:
                    continue
                d_pitch = min(abs(math.log2(float(f0s[i]) / float(f0s[j]))), PITCH_CAP) * PITCH_WEIGHT
                dist[i, j] = dist[j, i] = dist[i, j] + d_pitch
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def cluster_embeddings(
    embeddings: np.ndarray, k: int, threshold: float, f0s: list[float | None] | None = None
) -> tuple[np.ndarray, int]:
    """Assign each window a cluster label.

    threshold > 0: merge below the (combined) distance cut — may find MORE
    than k speakers (callers decide what to drop/merge; see diarize_media).
    threshold == 0: no collapsing, always exactly k clusters.
    Returns (labels, detected_speaker_count).
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), 0
    if n == 1:
        return np.zeros(1, dtype=int), 1

    from sklearn.cluster import AgglomerativeClustering

    if f0s is not None:
        dist = combined_distances(embeddings, f0s)

        def _fit(**kwargs):
            return AgglomerativeClustering(metric="precomputed", linkage="average", **kwargs).fit_predict(dist)

    else:

        def _fit(**kwargs):
            return AgglomerativeClustering(metric="cosine", linkage="average", **kwargs).fit_predict(
                embeddings
            )

    if threshold > 0:
        labels = _fit(n_clusters=None, distance_threshold=threshold)
        return labels, int(labels.max()) + 1
    return _fit(n_clusters=k), k


def merge_windows(
    windows: list[tuple[float, float]], labels, min_s: float = MIN_SEGMENT_S
) -> list[Segment]:
    """Merge consecutive same-cluster windows into segments, drop short ones.

    `labels` may contain None (unassigned windows, e.g. propagated neither
    way) — those windows are skipped and stay original audio.
    """
    segments: list[Segment] = []
    for (start, end), label in zip(windows, labels):
        if label is None:
            continue
        label = int(label)
        if segments and segments[-1].cluster == label and start - segments[-1].end < HOP_S + 1e-6:
            segments[-1].end = max(segments[-1].end, end)
        else:
            segments.append(Segment(start, end, label))
    return [s for s in segments if s.end - s.start >= min_s]


MAX_PROPAGATE_SLOTS = 3


def smooth_labels(labels: list, win_f0: list | None = None) -> list:
    """Temporal smoothing: flip isolated label blips to their neighbourhood.

    A window whose label differs from BOTH temporal neighbours, where those
    neighbours agree with each other, is usually a mid-sentence glitch
    (expressive pitch, overlapping speech) — real turn changes span multiple
    windows. But a genuine short interjection has the same temporal shape
    (A,A,B,A,A), so the flip is gated on pitch: the blip is only reassigned
    when its f0 sits closer (log2) to the flanking cluster's median than to
    its own cluster's median. Windows without a trusted f0 (and calls
    without any f0 at all) fall back to the temporal prior alone.
    """
    out: list[int | None] = list(labels)
    n = len(out)
    if n < 3:
        return out
    medians: dict[int, float | None] = {}
    solo: set[int] = set()
    if win_f0 is not None:
        for label in {x for x in out if x is not None}:
            vals = [f for f, x in zip(win_f0, out) if x == label and f is not None]
            medians[label] = float(np.median(vals)) if vals else None
    # A cluster whose ONLY pitch evidence is the blip itself carries no
    # identity (median == the blip's own f0) — treat it as unmeasured so the
    # temporal prior decides. Genuine interjections have siblings elsewhere.
    solo_labels = {
        label for label in medians
        if sum(1 for f, x in zip(win_f0 or [], out) if x == label and f is not None) <= 1
    }
    for i in range(1, n - 1):
        if out[i] is None:
            continue
        prev_l, next_l = out[i - 1], out[i + 1]
        if prev_l is None or next_l is None or prev_l != next_l or out[i] == prev_l:
            continue
        if not _pitch_supports_flip(win_f0, medians, i, out[i], prev_l,
                                    solo=(out[i] in solo_labels)):
            continue
        out[i] = prev_l
    return out


def _pitch_supports_flip(
    win_f0: list | None,
    medians: dict[int, float | None],
    i: int,
    own: int,
    flank: int,
    solo: bool = False,
) -> bool:
    """True when pitch evidence (or its absence) permits the flip."""
    if win_f0 is None:
        return True
    f0 = win_f0[i]
    if f0 is None or solo:
        return True  # untrusted pitch: temporal prior only
    own_median = medians.get(own)
    flank_median = medians.get(flank)
    if own_median is None or flank_median is None or own_median == flank_median:
        return True
    return abs(math.log2(f0 / flank_median)) < abs(math.log2(f0 / own_median))


def propagate_labels(windows: list[tuple[float, float]], labels: list) -> list:
    """Extend clustering to pitch-untrusted windows via temporal adjacency.

    None-labelled windows take the label of the nearest trusted window by
    slot distance; the nearest side wins a disagreement, and a tie (a
    window straddling a speaker change) or a nearest neighbour beyond
    MAX_PROPAGATE_SLOTS leaves the window unassigned — its audio stays
    original rather than guessing. Judged against originally-trusted labels
    only, so errors cannot cascade.
    """
    out: list[int | None] = list(labels)
    n = len(out)
    trusted = [i for i, l in enumerate(out) if l is not None]
    if not trusted:
        return out
    for i in range(n):
        if out[i] is not None:
            continue
        left = max((j for j in trusted if j < i), default=None)
        right = min((j for j in trusted if j > i), default=None)
        dl = i - left if left is not None else None
        dr = right - i if right is not None else None
        dl = dl if dl is not None and dl <= MAX_PROPAGATE_SLOTS else None
        dr = dr if dr is not None and dr <= MAX_PROPAGATE_SLOTS else None
        if dl is not None and dr is not None and left is not None and right is not None:
            if out[left] == out[right] or dl < dr:
                out[i] = out[left]
            elif dr < dl:
                out[i] = out[right]
        elif dl is not None and left is not None:
            out[i] = out[left]
        elif dr is not None and right is not None:
            out[i] = out[right]
    return out


def cluster_and_propagate(
    windows: list[tuple[float, float]],
    embeddings: np.ndarray,
    win_f0: list,
    k: int,
    threshold: float,
) -> tuple[list, int]:
    """Cluster the pitch-trusted windows (where the pitch feature applies),
    then propagate labels to untrusted windows by temporal adjacency.

    Returns (per-window labels with None for unassigned, detected count).
    Genuine speaker counts above k are force-merged down on the clustered
    subset (the combined distance keeps same-pitch speakers together).
    """
    from sklearn.cluster import AgglomerativeClustering

    n = len(windows)
    trusted = [i for i, f in enumerate(win_f0) if f is not None]
    use_pitch = len(trusted) >= 2
    idx = trusted if use_pitch else list(range(n))
    if not idx:
        return [None] * n, 0

    def _fit(n_clusters: int):
        """Cluster the trusted subset. `force=True` pins exactly n_clusters
        (the collapse threshold only ever bounds the result from above, so a
        fragmenting collapse can be re-pinned down to k)."""
        sub = embeddings[idx]
        if use_pitch:
            return AgglomerativeClustering(
                metric="precomputed", linkage="average",
                **({"n_clusters": None, "distance_threshold": threshold} if (threshold > 0 and not force_k)
                   else {"n_clusters": n_clusters}),
            ).fit_predict(combined_distances(sub, [win_f0[i] for i in idx]))
        return AgglomerativeClustering(
            metric="cosine", linkage="average",
            **({"n_clusters": None, "distance_threshold": threshold} if (threshold > 0 and not force_k)
               else {"n_clusters": n_clusters}),
        ).fit_predict(sub)

    def _spread(sub_labels):
        labels: list = [None] * n
        for i, l in zip(idx, sub_labels):
            labels[i] = int(l)
        if use_pitch:
            labels = propagate_labels(windows, labels)
        return labels

    force_k = False
    labels = smooth_labels(_spread(_fit(k)), win_f0)
    detected = len({l for l in labels if l is not None})
    if detected > k:
        # The expressive-pitch fragmentation case: re-cluster pinned to
        # exactly k so fragment sub-clusters merge back (combined distance
        # keeps same-pitch speakers together).
        force_k = True
        labels = smooth_labels(_spread(_fit(k)), win_f0)
        detected = k
    return labels, detected


def absorb_audible_gaps(
    audio16k: np.ndarray, segments: list[Segment], threshold_db: float = GAP_ABSORB_DB
) -> list[Segment]:
    """Extend segments to swallow audible gaps between them (in place of the
    input list — returns new Segment objects).

    Quieter in-between audio (laughter, reactions) sits below the VAD window
    threshold and would otherwise leak through as ORIGINAL audio even when
    every requested voice got a speaker. An audible gap (RMS within
    `threshold_db` of the media peak) joins the PRECEDING segment — laughter
    follows the joke. Silent gaps and leading/trailing edges stay original,
    so music intros/outros and true silence are preserved.
    """
    if len(segments) < 2:
        return [dataclasses.replace(s) for s in segments]
    peak_db = 20.0 * np.log10(float(np.abs(audio16k).max()) + 1e-10)

    def _audible(start: float, end: float) -> bool:
        chunk = slice_audio(audio16k, SAMPLE_RATE, start, end)
        if chunk.size == 0:
            return False
        rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(chunk**2))) + 1e-10)
        return rms_db > peak_db - threshold_db

    out = [dataclasses.replace(segments[0])]
    for seg in segments[1:]:
        if _audible(out[-1].end, seg.start):
            out[-1].end = seg.start  # absorb the gap into the previous speaker
        out.append(dataclasses.replace(seg))
    return out


# --- pitch ---------------------------------------------------------------------


def window_f0s(wavs: list[np.ndarray], f0_fn=measure_f0) -> list[float | None]:
    """Trusted per-window f0 (Hz), or None per window. Used as a clustering
    feature (male/female pairs that embeddings won't separate) and for the
    pitch spread inside a cluster."""
    return [f0_fn(w, SAMPLE_RATE) for w in wavs]


def cluster_f0s(
    audio16k: np.ndarray, segments: list[Segment], f0_fn=measure_f0
) -> dict[int, float | None]:
    """Median f0 per cluster, analyzing at most F0_ANALYSIS_S of its audio.

    Chunks never leave their segment's span — spilling into neighbouring
    segments would contaminate the pitch with another speaker's voice.
    """
    out: dict[int, float | None] = {}
    for cluster in sorted({s.cluster for s in segments}):
        budget = F0_ANALYSIS_S
        chunks: list[np.ndarray] = []
        for seg in (s for s in segments if s.cluster == cluster):
            if budget <= 0:
                break
            chunk = slice_audio(audio16k, SAMPLE_RATE, seg.start, seg.end)
            max_samples = int(budget * SAMPLE_RATE)
            if chunk.size > max_samples:
                chunk = chunk[:max_samples]
            chunks.append(chunk)
            budget -= chunk.size / SAMPLE_RATE
        merged = np.concatenate(chunks) if chunks else np.zeros(0)
        out[cluster] = f0_fn(merged, SAMPLE_RATE)
    return out


def assign_voices(
    f0s: dict[int, float | None],
    voices: list[str],
    voice_f0: dict[str, float | None],
    first_appearance: list[int],
) -> dict[int, str]:
    """Map clusters to voices preserving relative pitch order (rank-matching).

    When the counts of measured clusters and profiled voices are equal (the
    normal case), pairing is purely by rank: lowest-f0 cluster gets the
    lowest-f0 voice, and so on. Rank — not closest match — keeps the
    RELATIVE pitch between speakers intact even when both speakers sit
    outside the voices' pitch range (speakers 200/300Hz, voices 100/200Hz:
    rank gives 200->100, 300->200; closest-match would cross them).

    When the counts differ (a cluster too short to measure, a voice without
    a profile) ranks are ill-defined, so the measured subset falls back to
    greedy closest-match; whatever remains pairs up in first-appearance /
    config order. A cluster with no voice left over stays unmapped (its
    segments keep the original audio).
    """
    order = {c: i for i, c in enumerate(first_appearance)}
    voice_order = {v: i for i, v in enumerate(voices)}

    measured = sorted(
        (c for c in f0s if f0s[c] is not None), key=lambda c: (f0s[c], order[c])
    )
    profiled = sorted(
        (v for v in voices if voice_f0.get(v) is not None),
        key=lambda v: (voice_f0[v], voice_order[v]),
    )

    mapping: dict[int, str] = {}
    if len(measured) == len(profiled):
        # Full rank alignment: monotone in pitch by construction.
        mapping.update(zip(measured, profiled))
    else:
        # Partial information: closest-match the known subset...
        candidates = sorted(
            (
                (abs(math.log2(float(f0s[c]) / float(voice_f0[v]))), c, v)
                for c in measured
                for v in profiled
            ),
            key=lambda t: t[0],  # stable for ties: keep (cluster, voice) order
        )
        taken_clusters: set[int] = set()
        taken_voices: set[str] = set()
        for _, c, v in candidates:
            if c in taken_clusters or v in taken_voices:
                continue
            mapping[c] = v
            taken_clusters.add(c)
            taken_voices.add(v)
    # ...then leftovers (unmeasured clusters, unprofiled voices) in
    # appearance / config order.
    rest_clusters = [c for c in first_appearance if c not in mapping]
    rest_voices = [v for v in voices if v not in set(mapping.values())]
    mapping.update(zip(rest_clusters, rest_voices))
    return mapping


# --- embedding (model-backed, injectable in tests) -----------------------------


def embed_windows_wav2vec2(windows: list[np.ndarray], device: str = "cpu") -> np.ndarray:
    """Mean-pooled wav2vec2 features per window; loads the model lazily.

    Windows are 16kHz float32, at most WINDOW_S long; shorter tail windows
    are right-padded and pooled over their valid frames only (via lengths),
    so a truncated window is not diluted by padding. Batched to bound
    activation memory on CPU. Serialized by _MODEL_LOCK: a concurrent
    command waits instead of duplicating the model in RAM. `device` is
    evaluated where this runs (thin client: cpu; GPU server: cuda).
    """
    import gc

    import torch
    import torchaudio

    target = int(WINDOW_S * SAMPLE_RATE)
    padded = np.zeros((len(windows), target), dtype=np.float32)
    for i, w in enumerate(windows):
        n = min(w.size, target)
        padded[i, :n] = w[:n]
    # lengths must live on the model's device: extract_features builds the
    # attention bias from them (CPU lengths + CUDA model = device mismatch).
    lengths = torch.tensor(
        [min(w.size, target) for w in windows], dtype=torch.long, device=device
    )

    with _MODEL_LOCK:
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        model = bundle.get_model().to(device)
        model.eval()
        batch = torch.from_numpy(padded).to(device)
        with torch.inference_mode():
            feats_list, out_lengths = model.extract_features(batch, lengths=lengths)
        feats = feats_list[_FEATURE_LAYER]  # (batch, frames, 768)
        if out_lengths is not None:
            valid = torch.arange(feats.size(1))[None, :] < out_lengths[:, None].cpu()
            pooled = (
                (feats * valid[:, :, None].to(feats.device)).sum(dim=1)
                / out_lengths.to(feats.device).clamp(min=1)[:, None]
            )
        else:
            pooled = feats.mean(dim=1)
        # Release activations eagerly; the weights stay cached in torchaudio's
        # download dir but the ~400MB model is dropped from RAM after use.
        del feats_list, feats, batch, model
    pooled = pooled.cpu().numpy().astype(np.float32)
    gc.collect()
    return pooled


# --- top level ------------------------------------------------------------------


@dataclass
class DiarizationAnalysis:
    """Server-side half of diarization (Phase 2): timeline segments with per-
    cluster pitch, before voice assignment. JSON-serializable — the GPU
    server returns this; the thin client finalizes with its voice profiles.
    """

    segments: list[Segment]             # pre-gap-absorption timeline order
    cluster_f0: dict[int, float | None]  # kept clusters only
    detected: int


def diarize_media(
    path: str,
    voices: list[str],
    voice_f0: dict[str, float | None],
    threshold: float,
    device: str = "cpu",
    embed_fn=None,
    f0_fn=None,
) -> DiarizationResult:
    """Full pipeline: file -> timeline segments with a voice assigned to each.

    Raises on any failure; callers own error reporting.
    """
    analysis = analyze_media(path, len(voices), threshold, device=device,
                             embed_fn=embed_fn, f0_fn=f0_fn)
    return finalize_result(analysis, path, voices, voice_f0)


def analyze_media(
    path: str,
    num_voices: int,
    threshold: float,
    device: str = "cpu",
    embed_fn=None,
    f0_fn=None,
) -> DiarizationAnalysis:
    """File -> segments + per-cluster pitch (no voice assignment).

    The GPU-server half: everything here needs models/weights (wav2vec2,
    RMVPE); everything after it (rank-matched assignment, gap absorption)
    is pure math on data the thin client already has. `embed_fn`/`f0_fn`
    inject fakes in tests (called as fn(wavs_or_audio, sr)-shaped like the
    defaults); when omitted the real estimators run on `device`.
    Raises on any failure; callers own error reporting.
    """
    device = _resolve_device(device)
    embed = embed_fn or (lambda wavs: embed_windows_wav2vec2(wavs, device=device))
    pitch = f0_fn or (lambda a, sr: measure_f0(a, sr, device=device))

    audio, sr = load_mono(path)
    if audio.size < sr * 2:
        raise ValueError("media is too short to diarize")
    audio16k = resample(audio, sr, SAMPLE_RATE)

    mask = voice_mask(audio16k, SAMPLE_RATE)
    windows = voiced_windows(mask, audio16k.size / SAMPLE_RATE)
    if not windows:
        raise ValueError("no speech detected in media")
    log.info("Diarizing %d voiced windows (%.1fs media)", len(windows), audio16k.size / SAMPLE_RATE)

    wavs = [slice_audio(audio16k, SAMPLE_RATE, s, e) for s, e in windows]
    embeddings = embed(wavs)
    k = num_voices
    # Per-window pitch: trusted windows drive the clustering distance (it
    # separates pitch-different speakers the embeddings interleave, e.g.
    # male+female); untrusted windows get labels by temporal adjacency.
    win_f0 = window_f0s(wavs, f0_fn=pitch)
    labels, detected = cluster_and_propagate(windows, embeddings, win_f0, k, threshold)

    # Negligible clusters (a breath, a noise burst at the clip edge) are not
    # speakers — they keep original audio instead of consuming a voice. This
    # runs BEFORE the force-merge so a >k collapse with trailing noise does
    # not push a real speaker's voice onto the noise.
    min_speaker = max(MIN_SPEAKER_S, MIN_SPEAKER_FRAC * audio16k.size / SAMPLE_RATE)
    durations: dict[int, float] = {}
    for (start, end), label in zip(windows, labels):
        if label is None:
            continue
        durations[label] = durations.get(label, 0.0) + end - start
    kept = {c for c, d in durations.items() if d >= min_speaker}
    dropped = sorted(set(durations) - kept)
    if dropped and len(kept) >= 1:
        log.info(
            "Ignoring noise cluster(s) %s (< %.1fs of speech); they stay original audio",
            dropped, min_speaker,
        )
        labels = [l if l in kept else None for l in labels]
    if len(kept) > k:
        # Genuine 3+ speakers for k voices: merge the closest ones (the
        # pitch-combined distance keeps the assignment meaningful).
        labels, _ = cluster_and_propagate(windows, embeddings, win_f0, k, 0.0)
        detected = k
        kept = set(range(k))
    else:
        detected = len(kept)
    if not kept:
        raise ValueError("no sustained speech found (all speech runs too short)")
    log.info(
        "Clustering: %d speaker(s) detected for %d voice(s) (threshold=%s)",
        detected, k, threshold or "off",
    )

    segments = merge_windows(windows, labels)
    if not segments:
        raise ValueError("speech runs were too short to segment")

    f0s = cluster_f0s(audio16k, segments, f0_fn=pitch)
    f0s = {c: f for c, f in f0s.items() if c in kept}
    return DiarizationAnalysis(segments=segments, cluster_f0=f0s, detected=detected)


def finalize_result(
    analysis: DiarizationAnalysis,
    source_path: str,
    voices: list[str],
    voice_f0: dict[str, float | None],
) -> DiarizationResult:
    """Voice assignment + gap absorption for a server (or local) analysis.

    Pure math + the source file the thin client already has: rank-matched
    voice mapping from its pitch profiles, then audible-gap swallowing
    (which runs AFTER pitch analysis so laughter never skews the medians).
    """
    f0s = analysis.cluster_f0
    first_appearance = list(dict.fromkeys(s.cluster for s in analysis.segments))
    mapping = assign_voices(f0s, voices, voice_f0, first_appearance)
    audio, sr = load_mono(source_path)
    audio16k = resample(audio, sr, SAMPLE_RATE)
    # Swallow audible gaps (laughter, reactions) into the preceding speaker.
    # The conversion slices must cover the gaps so they get converted
    # instead of leaking through as original audio.
    segments = absorb_audible_gaps(audio16k, analysis.segments)
    log.info(
        "Voice assignment: %s",
        ", ".join(
            f"speaker {c} ({f0s[c] and round(f0s[c]) or '?'}Hz) -> {v}"
            for c, v in mapping.items()
        ),
    )
    return DiarizationResult(
        segments=segments,
        voice_of_cluster=mapping,
        cluster_f0=f0s,
        detected=analysis.detected,
    )


def result_summary(result: DiarizationResult) -> str:
    """Human-readable assignment line for the done status."""
    parts = []
    for cluster, voice in result.voice_of_cluster.items():
        f0 = result.cluster_f0.get(cluster)
        parts.append(f"{voice} @ {round(f0)}Hz" if f0 else f"{voice} @ ?Hz")
    return ", ".join(parts)


def rebuild_timeline(
    source_path: str,
    result: DiarizationResult,
    converted_paths: list,
    out_path: str,
) -> None:
    """Reassemble the full media at the source sample rate.

    Converted segments go back in their original places; everything between
    them (music, silence, effects) is kept as the ORIGINAL audio — it passes
    through unconverted, which both sounds right and skips wasted RVC work.
    A `None` entry in `converted_paths` (cluster left unassigned) keeps the
    original audio for that segment.
    """
    orig, sr = load_mono(source_path)
    duration = orig.size / sr
    parts: list[np.ndarray] = []
    cursor = 0.0
    for i, seg in enumerate(result.segments):
        if seg.start > cursor + 1e-3:
            parts.append(slice_audio(orig, sr, cursor, seg.start))
        path = converted_paths[i] if i < len(converted_paths) else None
        if path is not None:
            converted, conv_sr = load_mono(path)
            parts.append(resample(converted, conv_sr, sr))
        else:
            parts.append(slice_audio(orig, sr, seg.start, seg.end))
        cursor = max(cursor, seg.end)
    if cursor < duration - 1e-3:
        parts.append(slice_audio(orig, sr, cursor, duration))
    write_wav(out_path, concat_audio(parts, sr=sr), sr)


# --- voice pitch profiles (written by tools/analyze_voices.py) ------------------


def load_profiles(path) -> dict[str, dict]:
    import json

    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_profiles(path, profiles: dict[str, dict]) -> None:
    import json

    with open(path, "w") as f:
        json.dump(profiles, f, indent=2, sort_keys=True)


def profile_key(voice_cfg, model_mtime: float) -> dict:
    """The tuning fingerprint a profile is valid for. Includes the f0
    estimator: profile values are only comparable when measured the same
    way (pyin and rmvpe disagree by octaves on ambiguous audio)."""
    return {
        "model": str(voice_cfg.rvc_model),
        "mtime": model_mtime,
        "pitch": voice_cfg.pitch,
        "f0_method": voice_cfg.f0_method,
        "estimator": "rmvpe",
    }


def profile_f0(profiles: dict[str, dict], name: str, voice_cfg) -> float | None:
    """Cached f0 for a voice, or None when missing/stale (model changed)."""
    entry = profiles.get(name)
    if not isinstance(entry, dict) or "f0" not in entry:
        return None
    key = entry.get("key") or {}
    expected = {"model": str(voice_cfg.rvc_model), "pitch": voice_cfg.pitch,
                "f0_method": voice_cfg.f0_method, "estimator": "rmvpe"}
    for k, v in expected.items():
        if key.get(k) != v:
            return None
    stored_mtime = key.get("mtime")
    if stored_mtime is not None:
        from pathlib import Path

        try:
            if Path(voice_cfg.rvc_model).stat().st_mtime != stored_mtime:
                return None
        except OSError:
            return None
    f0 = entry.get("f0")
    return float(f0) if f0 else None


def voice_f0_map(profiles: dict[str, dict], voices: list) -> dict[str, float | None]:
    """Bulk profile lookup used by the bot; voices are VoiceConfig objects."""
    out: dict[str, float | None] = {}
    for cfg in voices:
        out[cfg.name] = profile_f0(profiles, cfg.name, cfg)
    return out


__all__ = [
    "DiarizationAnalysis",
    "DiarizationResult",
    "Segment",
    "absorb_audible_gaps",
    "analyze_media",
    "assign_voices",
    "cluster_embeddings",
    "cluster_f0s",
    "diarize_media",
    "embed_windows_wav2vec2",
    "finalize_result",
    "load_profiles",
    "measure_f0",
    "merge_windows",
    "profile_f0",
    "profile_key",
    "rebuild_timeline",
    "result_summary",
    "save_profiles",
    "voice_f0_map",
    "voice_mask",
    "voiced_windows",
]
