"""A dependency-light spectral-centroid tag adapter (works on synthetic bursts)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.adapter import ModelAdapter
from ..core.data import Prediction, Sample
from ..core.registry import ADAPTERS


@ADAPTERS.register("dsp_tag")
class DSPTagAdapter(ModelAdapter):
    """Tags a clip low/high by spectral centroid of the first second.

    Returns soft scores (0..1) so the mAP/AUC metrics have meaningful ranks.
    """

    name = "dsp_tag"

    def __init__(self, low_hz: float = 150.0, high_hz: float = 2000.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)

    def load(self) -> None:
        return None

    def predict(self, sample: Sample) -> Prediction:
        from scipy.io import wavfile

        sr, data = wavfile.read(sample.audio_path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        else:
            data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        # Use the first second (where the first bursts live).
        seg = data[: min(len(data), sr)]
        centroid = _spectral_centroid(seg, sr)
        # Distance of centroid to low vs high center -> soft scores.
        d_low = abs(centroid - self.low_hz)
        d_high = abs(centroid - self.high_hz)
        total = d_low + d_high + 1e-9
        score_high = d_low / total
        score_low = d_high / total
        return Prediction(
            id=sample.id,
            output={"scores": {"low": float(score_low), "high": float(score_high)}},
        )


def _spectral_centroid(seg: np.ndarray, sr: int) -> float:
    n = len(seg)
    if n == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    denom = spec.sum()
    if denom < 1e-12:
        return 0.0
    return float((freqs * spec).sum() / denom)
