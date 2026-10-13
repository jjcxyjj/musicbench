"""Synthetic dataset for smoke-testing the harness end-to-end.

Generates short "music-like" clips whose ground truth is fully known:

* **beats/tempo**: a regular train of tone bursts whose onset times are the
  reference beats (tempo known exactly).
* **tags**: each clip is either "low" (bursts at ~150 Hz) or "high" (bursts at
  ~2000 Hz), giving a two-tag vocabulary with clean positives/negatives.

This lets the tests verify the pipeline deterministically without downloading
GTZAN or any external dataset. ``name`` in config should be ``synthetic``.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List

import numpy as np

from ..core.data import Sample
from ..core.dataset import Dataset
from ..core.registry import DATASETS


@DATASETS.register("synthetic")
class SyntheticDataset(Dataset):
    name = "synthetic"

    def __init__(
        self,
        num_samples: int = 8,
        sr: int = 22050,
        duration: float = 6.0,
        tempo: float = 120.0,
        seed: int = 0,
    ) -> None:
        self.num_samples = int(num_samples)
        self.sr = int(sr)
        self.duration = float(duration)
        self.tempo = float(tempo)
        self.seed = int(seed)

    # ------------------------------------------------------------------ load
    def load(self, split: str = "test") -> List[Sample]:
        rng = np.random.default_rng(self.seed)
        workdir = tempfile.mkdtemp(prefix="musicbench-synthetic-")
        samples: List[Sample] = []
        for i in range(self.num_samples):
            is_high = (i % 2) == 0  # half low, half high -> both classes per tag
            sample = self._make_sample(i, is_high, workdir, rng)
            samples.append(sample)
        return samples

    # ------------------------------------------------------------- synthesis
    def _make_sample(self, idx: int, is_high: bool, workdir: str, rng: np.random.Generator) -> Sample:
        sr = self.sr
        tempo = self.tempo
        beat_interval = 60.0 / tempo
        # First beat at 0.5s, then one per interval; last beat <= duration - 0.2.
        beats = list(np.arange(0.5, self.duration - 0.2, beat_interval))

        freq = 2000.0 if is_high else 150.0
        audio = self._render_bursts(beats, freq, sr, self.duration, rng)

        path = os.path.join(workdir, f"sample_{idx:04d}.wav")
        self._write_wav(path, audio, sr)

        tags = {"low": int(not is_high), "high": int(is_high)}
        return Sample(
            id=f"synth_{idx:04d}",
            audio_path=path,
            metadata={"sr": sr, "duration": self.duration, "is_high": bool(is_high)},
            ground_truth={"beats": beats, "tempo": tempo, "tags": tags},
        )

    def _render_bursts(
        self,
        beats: List[float],
        freq: float,
        sr: int,
        duration: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n = int(sr * duration)
        t = np.arange(n) / sr
        signal = np.zeros(n, dtype=np.float32)
        burst_len = int(0.08 * sr)
        for b in beats:
            start = int(b * sr)
            end = min(start + burst_len, n)
            if start >= n:
                break
            # Slight amplitude/phase jitter keeps onset detection honest but
            # the underlying beat grid stays exactly at ``beats``.
            env = np.hanning(end - start).astype(np.float32)
            amp = float(rng.uniform(0.8, 1.0))
            seg = t[start:end] - b
            signal[start:end] += (amp * env * np.sin(2 * np.pi * freq * seg)).astype(np.float32)
        # Normalize to [-1, 1].
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak
        return signal

    def _write_wav(self, path: str, audio: np.ndarray, sr: int) -> None:
        from scipy.io import wavfile

        wavfile.write(path, sr, audio)


def synthetic_sample_dict(idx: int = 0, **overrides: Any) -> Dict[str, Any]:
    """Helper returning a sample as a plain dict for tests/adapters."""
    ds = SyntheticDataset(num_samples=idx + 1)
    sample = ds.load("test")[idx]
    return {"id": sample.id, "audio_path": sample.audio_path, "ground_truth": sample.ground_truth}
