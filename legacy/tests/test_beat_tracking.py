"""Beat tracking task tests: metric alignment with mir_eval + pipeline sanity."""

import numpy as np
import pytest

import musicbench.tasks  # noqa: F401
import musicbench.metrics  # noqa: F401
import musicbench.datasets  # noqa: F401
import musicbench.adapters  # noqa: F401

from musicbench.core.data import Prediction, Sample
from musicbench.core.registry import get_metric


def _make_samples_and_preds():
    """Perfect predictions -> F-measure should be 1.0."""
    beats = np.arange(0.5, 6.0, 0.5).tolist()  # 0.5..5.5 at 120 BPM
    tempo = 120.0
    samples = [
        Sample(id="a", audio_path="unused.wav", ground_truth={"beats": beats, "tempo": tempo})
    ]
    preds = [Prediction(id="a", output={"beats": beats, "tempo": tempo})]
    return samples, preds


def test_beat_fmeasure_perfect_alignment():
    samples, preds = _make_samples_and_preds()
    metric = get_metric("beat_fmeasure")()
    out = metric.compute(preds, samples)
    assert out["n_evaluated"] == 1
    assert out["f_measure"] == pytest.approx(1.0, abs=1e-6)


def test_tempo_mae_perfect():
    samples, preds = _make_samples_and_preds()
    metric = get_metric("tempo_mae")()
    out = metric.compute(preds, samples)
    assert out["tempo_mae"] == pytest.approx(0.0, abs=1e-6)


def test_beat_fmeasure_missing_half_beats_degrades():
    """Dropping half the beats should lower (but not zero) the F-measure."""
    beats = np.arange(0.5, 6.0, 0.5).tolist()
    half = np.asarray(beats)[::2].tolist()
    samples = [Sample(id="a", audio_path="u", ground_truth={"beats": beats, "tempo": 120.0})]
    preds = [Prediction(id="a", output={"beats": half, "tempo": 120.0})]
    metric = get_metric("beat_fmeasure")()
    out = metric.compute(preds, samples)
    assert 0.0 < out["f_measure"] < 1.0


def test_dsp_beat_adapter_end_to_end_on_synthetic():
    """The DSP adapter should recover beats/tempo from synthetic bursts well."""
    from musicbench.adapters.beat import DSPBeatAdapter
    from musicbench.datasets.synthetic import SyntheticDataset

    samples = SyntheticDataset(num_samples=4, seed=0).load("test")
    adapter = DSPBeatAdapter()
    adapter.load()
    preds = [adapter.predict(s) for s in samples]

    fm = get_metric("beat_fmeasure")()
    tm = get_metric("tempo_mae")()
    f_out = fm.compute(preds, samples)
    t_out = tm.compute(preds, samples)

    assert f_out["f_measure"] > 0.7, f"expected good beat recovery, got {f_out}"
    assert t_out["tempo_mae"] < 10.0, f"expected accurate tempo, got {t_out}"
