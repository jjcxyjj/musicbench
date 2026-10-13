"""L0 "basic" audio quality metrics for text-to-music (no deep models).

Compares each generated audio against its reference audio on per-sample
objective features and reports mean absolute error for each feature. Lower is
better for every feature below.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from ..core.audio_features import extract_all
from ..core.data import Prediction, Sample
from ..core.metric import Metric
from ..core.registry import METRICS

FEATURES = ["duration", "rms_db", "spectral_centroid", "zero_crossing_rate", "crest_factor", "silence_ratio"]


@METRICS.register("audio_quality")
class AudioQualityMetric(Metric):
    """Mean absolute error of objective features: generated vs reference audio.

    The prediction output must contain ``audio_path`` (the generated file).
    """

    name = "audio_quality"

    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        by_id = {s.id: s for s in samples}
        errs: Dict[str, List[float]] = {f: [] for f in FEATURES}
        n = 0
        for p in predictions:
            s = by_id.get(p.id)
            gen_path = p.output.get("audio_path") if isinstance(p.output, dict) else None
            if s is None or not gen_path or not os.path.isfile(gen_path):
                continue
            if not os.path.isfile(s.audio_path):
                continue
            try:
                ref = extract_all(s.audio_path)
                gen = extract_all(gen_path)
            except Exception:
                continue
            n += 1
            for f in FEATURES:
                errs[f].append(abs(float(gen[f]) - float(ref[f])))

        out: Dict[str, Any] = {"n_evaluated": n, "available": n > 0,
                               "coverage": n / len(samples) if samples else 0.0}
        for f in FEATURES:
            out[f"mae_{f}"] = float(np.mean(errs[f])) if errs[f] else float("nan")
        return out
