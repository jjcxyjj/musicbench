"""Tagging task tests: mAP / macro F1 / macro AUC on clean separable scores."""

import numpy as np
import pytest

import musicbench.tasks  # noqa: F401
import musicbench.metrics  # noqa: F401
import musicbench.datasets  # noqa: F401
import musicbench.adapters  # noqa: F401

from musicbench.core.data import Prediction, Sample
from musicbench.core.registry import get_metric


def _perfect_tagging_set():
    """Two tags (low/high), 4 samples, perfectly separable soft scores."""
    samples = []
    preds = []
    for i, is_high in enumerate([True, True, False, False]):
        tags = {"low": int(not is_high), "high": int(is_high)}
        samples.append(Sample(id=f"s{i}", audio_path="u", ground_truth={"tags": tags}))
        if is_high:
            scores = {"low": 0.0, "high": 1.0}
        else:
            scores = {"low": 1.0, "high": 0.0}
        preds.append(Prediction(id=f"s{i}", output={"scores": scores}))
    return samples, preds


def test_tag_map_perfect():
    samples, preds = _perfect_tagging_set()
    out = get_metric("tag_map")().compute(preds, samples)
    assert out["map"] == pytest.approx(1.0, abs=1e-6)


def test_tag_macro_f1_perfect():
    samples, preds = _perfect_tagging_set()
    out = get_metric("tag_macro_f1")().compute(preds, samples)
    assert out["macro_f1"] == pytest.approx(1.0, abs=1e-6)


def test_tag_macro_auc_perfect():
    samples, preds = _perfect_tagging_set()
    out = get_metric("tag_macro_auc")().compute(preds, samples)
    assert out["macro_auc"] == pytest.approx(1.0, abs=1e-6)


def test_dsp_tag_adapter_end_to_end_on_synthetic():
    from musicbench.adapters.tagging import DSPTagAdapter
    from musicbench.datasets.synthetic import SyntheticDataset

    samples = SyntheticDataset(num_samples=8, seed=0).load("test")
    adapter = DSPTagAdapter()
    adapter.load()
    preds = [adapter.predict(s) for s in samples]

    map_out = get_metric("tag_map")().compute(preds, samples)
    f1_out = get_metric("tag_macro_f1")().compute(preds, samples)
    auc_out = get_metric("tag_macro_auc")().compute(preds, samples)

    assert map_out["map"] > 0.9, f"expected near-perfect mAP, got {map_out}"
    assert f1_out["macro_f1"] > 0.9, f"expected near-perfect F1, got {f1_out}"
    assert auc_out["macro_auc"] > 0.9, f"expected near-perfect AUC, got {auc_out}"
