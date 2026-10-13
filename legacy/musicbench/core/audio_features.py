"""Dependency-light audio feature extraction (scipy/numpy only).

These power the L0 "basic" metrics: no deep models, no torch. They are
per-sample scalar features used to compare generated audio against reference
audio for text-to-music evaluation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def load_mono(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load an audio file as a float32 mono array in [-1, 1].

    If ``target_sr`` is given, the signal is resampled to it (scipy.signal).
    """
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if target_sr is not None and target_sr != sr:
        gcd = int(np.gcd(sr, target_sr))
        data = resample_poly(data, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return data, sr


def duration_seconds(x: np.ndarray, sr: int) -> float:
    return float(len(x) / sr)


def rms_db(x: np.ndarray, eps: float = 1e-10) -> float:
    """RMS level in dBFS."""
    rms = float(np.sqrt(np.mean(x ** 2)) + eps)
    return float(20.0 * np.log10(rms + 1e-12))


def spectral_centroid(x: np.ndarray, sr: int) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    denom = spec.sum()
    if denom < 1e-12:
        return 0.0
    return float((freqs * spec).sum() / denom)


def zero_crossing_rate(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    sign = np.sign(x)
    crossings = np.abs(np.diff(sign)).sum() / 2.0
    return float(crossings / len(x))


def crest_factor(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    return float(peak / rms)


def silence_ratio(x: np.ndarray, sr: int, threshold_db: float = -60.0, hop_sec: float = 0.1) -> float:
    """Fraction of frames whose RMS falls below ``threshold_db``."""
    hop = max(1, int(hop_sec * sr))
    frame = max(hop, 2048)
    if len(x) < frame:
        return 1.0 if rms_db(x) < threshold_db else 0.0
    silent = 0
    total = 0
    for start in range(0, len(x) - frame + 1, hop):
        seg = x[start:start + frame]
        if rms_db(seg) < threshold_db:
            silent += 1
        total += 1
    return float(silent / total) if total else 0.0


def extract_all(path: str, target_sr: Optional[int] = 22050) -> dict:
    """Return a dict of all L0 features for one audio file."""
    x, sr = load_mono(path, target_sr)
    return {
        "duration": duration_seconds(x, sr),
        "rms_db": rms_db(x),
        "spectral_centroid": spectral_centroid(x, sr),
        "zero_crossing_rate": zero_crossing_rate(x),
        "crest_factor": crest_factor(x),
        "silence_ratio": silence_ratio(x, sr),
    }
