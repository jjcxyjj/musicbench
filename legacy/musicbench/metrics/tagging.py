"""Tagging metrics via sklearn: mAP, macro F1, macro AUC."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..core.data import Prediction, Sample
from ..core.metric import Metric
from ..core.registry import METRICS


def _build_matrices(predictions: List[Prediction], samples: List[Sample]):
    """Return (y_true, y_score, tags) aligned over the tag vocabulary.

    y_true is 0/1 multi-label; y_score is soft scores in [0,1]. Both are
    (n_samples, n_tags) arrays. Rows are matched by sample id; missing ids or
    missing scores are dropped.
    """
    by_id = {s.id: s for s in samples}
    tags: List[str] = []
    for s in samples:
        for t in s.ground_truth.get("tags", {}).keys():
            if t not in tags:
                tags.append(t)
    for p in predictions:
        for t in p.output.get("scores", {}).keys():
            if t not in tags:
                tags.append(t)

    rows_true: List[List[int]] = []
    rows_score: List[List[float]] = []
    for p in predictions:
        s = by_id.get(p.id)
        if s is None:
            continue
        gt = s.ground_truth.get("tags", {})
        sc = p.output.get("scores", {})
        if not sc:
            continue
        rows_true.append([1 if gt.get(t) else 0 for t in tags])
        rows_score.append([float(sc.get(t, 0.0)) for t in tags])

    if not rows_true:
        return None, None, tags
    return np.asarray(rows_true, dtype=int), np.asarray(rows_score, dtype=float), tags


@METRICS.register("tag_map")
class TagMAP(Metric):
    """Mean average precision (multi-label, macro over tags)."""

    name = "tag_map"

    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        from sklearn.metrics import average_precision_score

        y_true, y_score, _ = _build_matrices(predictions, samples)
        if y_true is None:
            return {"map": float("nan")}
        try:
            return {"map": float(average_precision_score(y_true, y_score, average="macro"))}
        except Exception:
            return {"map": float("nan")}


@METRICS.register("tag_macro_f1")
class TagMacroF1(Metric):
    """Macro F1 with scores binarized at 0.5."""

    name = "tag_macro_f1"

    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        from sklearn.metrics import f1_score

        y_true, y_score, _ = _build_matrices(predictions, samples)
        if y_true is None:
            return {"macro_f1": float("nan")}
        y_pred = (y_score >= 0.5).astype(int)
        try:
            return {"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}
        except Exception:
            return {"macro_f1": float("nan")}


@METRICS.register("tag_macro_auc")
class TagMacroAUC(Metric):
    """Macro ROC-AUC over tags (requires each tag to have both classes present)."""

    name = "tag_macro_auc"

    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        from sklearn.metrics import roc_auc_score

        y_true, y_score, _ = _build_matrices(predictions, samples)
        if y_true is None:
            return {"macro_auc": float("nan")}
        try:
            return {"macro_auc": float(roc_auc_score(y_true, y_score, average="macro"))}
        except Exception:
            return {"macro_auc": float("nan")}
