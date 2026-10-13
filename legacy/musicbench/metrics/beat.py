"""Beat tracking metrics via ``mir_eval.beat``."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..core.data import Prediction, Sample
from ..core.metric import Metric
from ..core.registry import METRICS


@METRICS.register("beat_fmeasure")
class BeatFMeasure(Metric):
    """mir_eval.beat F-measure (with the standard 0.07 tolerance window).

    Also reports Cemgil and P-score from the same evaluation, since they come
    for free and are the de-facto companions of beat F-measure.
    """

    name = "beat_fmeasure"

    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        import mir_eval.beat

        by_id = {s.id: s for s in samples}
        f_scores: List[float] = []
        cemgil: List[float] = []
        p_score: List[float] = []
        for p in predictions:
            sample = by_id.get(p.id)
            if sample is None:
                continue
            ref = np.asarray(sample.ground_truth.get("beats", []), dtype=float)
            est = np.asarray(p.output.get("beats", []), dtype=float)
            if ref.size == 0 or est.size == 0:
                # mir_eval warns and returns zeros on empty; skip to keep clean.
                continue
            try:
                scores = mir_eval.beat.evaluate(ref, est)
                f_scores.append(float(scores["F-measure"]))
                cemgil.append(float(scores["Cemgil"]))
                p_score.append(float(scores["P-score"]))
            except Exception:
                continue
        return {
            "f_measure": float(np.mean(f_scores)) if f_scores else float("nan"),
            "cemgil": float(np.mean(cemgil)) if cemgil else float("nan"),
            "p_score": float(np.mean(p_score)) if p_score else float("nan"),
            "n_evaluated": len(f_scores),
        }


@METRICS.register("tempo_mae")
class TempoMAE(Metric):
    """Mean absolute error of estimated tempo (BPM) vs. reference tempo."""

    name = "tempo_mae"

    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        by_id = {s.id: s for s in samples}
        errs: List[float] = []
        for p in predictions:
            sample = by_id.get(p.id)
            if sample is None:
                continue
            ref = sample.ground_truth.get("tempo")
            est = p.output.get("tempo")
            if ref is None or est is None:
                continue
            try:
                errs.append(abs(float(est) - float(ref)))
            except (TypeError, ValueError):
                continue
        return {
            "tempo_mae": float(np.mean(errs)) if errs else float("nan"),
            "n_evaluated": len(errs),
        }
