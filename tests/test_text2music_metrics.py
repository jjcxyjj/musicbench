"""Tests for the L1 professional metrics: VGGish-based FAD and SongEval.

These tests exercise the pure-math path and the graceful-degradation path.
They do NOT require torch / torchvggish / muq to be installed.
"""

import os

import numpy as np
import pytest

import musicbench.tasks  # noqa: F401
import musicbench.metrics  # noqa: F401

from musicbench.core.data import Prediction, Sample
from musicbench.core.registry import get_metric


def _write_wav(path, sr=22050, freq=440.0, dur=1.0):
    from scipy.io import wavfile

    t = np.arange(int(sr * dur)) / sr
    x = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    wavfile.write(path, sr, x)
    return path


# ------------------------------------------------------------------ FAD math
def test_fad_zero_for_identical_embeddings():
    from musicbench.metrics.generation import _fad

    rng = np.random.RandomState(0)
    a = rng.randn(20, 8)
    assert _fad(a, a) == pytest.approx(0.0, abs=1e-8)


def test_fad_positive_for_different_embeddings():
    from musicbench.metrics.generation import _fad

    rng = np.random.RandomState(0)
    a = rng.randn(20, 8)
    b = rng.randn(20, 8) + 5.0
    assert _fad(a, b) > 0.0


# ------------------------------------------------------------------ FAD graceful degradation
def test_fad_degrades_gracefully_without_torchvggish(tmp_path):
    ref = _write_wav(str(tmp_path / "ref.wav"))
    gen = _write_wav(str(tmp_path / "gen.wav"))
    samples = [Sample(id="s0", audio_path=ref), Sample(id="s1", audio_path=ref)]
    preds = [Prediction(id="s0", output={"audio_path": gen}),
             Prediction(id="s1", output={"audio_path": gen})]
    out = get_metric("fad")().compute(preds, samples)
    # torchvggish is not installed in this test env -> graceful degradation.
    assert out["available"] is False


# ------------------------------------------------------------------ SongEval graceful degradation
def test_songeval_degrades_gracefully_without_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSICBENCH_SONGEVAL_CKPT", str(tmp_path / "missing.safetensors"))
    gen = _write_wav(str(tmp_path / "gen.wav"))
    samples = [Sample(id="s0", audio_path=str(tmp_path / "ref.wav"))]
    preds = [Prediction(id="s0", output={"audio_path": gen})]
    out = get_metric("songeval")().compute(preds, samples)
    assert out["available"] is False
    assert "reason" in out


def test_songeval_reports_all_five_dimensions():
    from musicbench.metrics.songeval import DIMENSIONS

    assert DIMENSIONS == ["coherence", "musicality", "memorability",
                          "clarity", "naturalness"]


# ------------------------------------------------------------------ ranking wiring
def test_ranking_songeval_higher_is_better():
    from musicbench.core.ranking import rank

    scores = {
        "a": {"songeval": {"coherence": 4.5, "mean": 4.4}},
        "b": {"songeval": {"coherence": 1.2, "mean": 1.1}},
    }
    r = rank(scores)
    assert r[0]["algorithm"] == "a"