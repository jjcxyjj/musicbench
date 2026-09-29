"""Tests for the text2music additions: features, feature_json dataset, ranking,
and L0 audio_quality metric."""

import json
import os

import numpy as np
import pytest

import musicbench.tasks  # noqa: F401
import musicbench.metrics  # noqa: F401
import musicbench.datasets  # noqa: F401
import musicbench.adapters  # noqa: F401

from musicbench.core.data import Prediction, Sample
from musicbench.core.ranking import rank
from musicbench.core.registry import get_metric


# ------------------------------------------------------------------ fixtures
def _write_wav(path, sr=22050, freq=440.0, dur=1.0):
    from scipy.io import wavfile

    t = np.arange(int(sr * dur)) / sr
    x = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    wavfile.write(path, sr, x)
    return path


def _make_feature_json_dir(tmp_path, n=12):
    """Create a fake /data/MusicLite_data/feature/<batch>/5_with_latent.json."""
    root = tmp_path / "feature"
    batch = root / "20260909"
    audio_dir = batch / "audio"
    audio_dir.mkdir(parents=True)
    records = []
    for i in range(n):
        wav = _write_wav(str(audio_dir / f"chunk{i}.wav"), freq=440.0 + i * 50)
        records.append({
            "path": wav,
            "has_vocal": i % 2 == 0,
            "vocal_score": 0.2 + i * 0.01,
            "Qwen_caption": f"sample caption {i}",
            "gf_style_name": "国风流行",
            "gf_style_code": "gflx",
            "path_original": wav,
        })
    batch.joinpath("5_with_latent.json").write_text(json.dumps(records))
    return root


# ------------------------------------------------------------------ feature_json
def test_feature_json_samples_random_N(tmp_path):
    root = _make_feature_json_dir(tmp_path, n=12)
    from musicbench.datasets.feature_json import FeatureJsonDataset

    ds = FeatureJsonDataset(root=str(root), num_samples=5, seed=0)
    samples = ds.load("test")
    assert len(samples) == 5
    for s in samples:
        assert s.metadata["caption"].startswith("sample caption")
        assert s.metadata["style_name"] == "国风流行"
        assert os.path.isfile(s.audio_path)


def test_feature_json_deterministic_with_seed(tmp_path):
    root = _make_feature_json_dir(tmp_path, n=12)
    from musicbench.datasets.feature_json import FeatureJsonDataset

    ids1 = [s.id for s in FeatureJsonDataset(root=str(root), num_samples=5, seed=7).load()]
    ids2 = [s.id for s in FeatureJsonDataset(root=str(root), num_samples=5, seed=7).load()]
    assert ids1 == ids2


# ------------------------------------------------------------------ audio_features
def test_audio_features_on_sine(tmp_path):
    from musicbench.core.audio_features import extract_all

    wav = _write_wav(str(tmp_path / "sine.wav"), freq=440.0, dur=1.0)
    feats = extract_all(wav)
    assert feats["duration"] == pytest.approx(1.0, abs=0.02)
    assert feats["spectral_centroid"] > 300  # 440Hz sine
    assert feats["silence_ratio"] == pytest.approx(0.0, abs=0.01)


# ------------------------------------------------------------------ audio_quality metric
def test_audio_quality_perfect_when_identical(tmp_path):
    wav = _write_wav(str(tmp_path / "a.wav"), freq=440.0, dur=1.0)
    samples = [Sample(id="s0", audio_path=wav)]
    preds = [Prediction(id="s0", output={"audio_path": wav})]
    out = get_metric("audio_quality")().compute(preds, samples)
    assert out["n_evaluated"] == 1
    for k in ["mae_duration", "mae_rms_db", "mae_spectral_centroid"]:
        assert out[k] == pytest.approx(0.0, abs=1e-3)


# ------------------------------------------------------------------ ranking
def test_ranking_picks_best_algorithm():
    scores = {
        "good": {"audio_quality": {"mae_duration": 0.1, "mae_rms_db": 1.0}},
        "bad": {"audio_quality": {"mae_duration": 2.0, "mae_rms_db": 20.0}},
    }
    r = rank(scores)
    assert r[0]["algorithm"] == "good"
    assert r[0]["score"] > r[1]["score"]


def test_ranking_skips_unavailable_metric():
    scores = {
        "a": {"audio_quality": {"mae_duration": 0.5}, "fad": {"available": False, "fad": float("nan")}},
        "b": {"audio_quality": {"mae_duration": 1.5}, "fad": {"available": False, "fad": float("nan")}},
    }
    r = rank(scores)
    # fad is NaN everywhere -> skipped, only audio_quality used
    assert all("fad.fad" in item["skipped_metrics"] for item in r)
    assert r[0]["algorithm"] == "a"


def test_ranking_higher_better_metric():
    # clap_score is higher-is-better
    scores = {
        "a": {"clap_score": {"clap": 0.9}},
        "b": {"clap_score": {"clap": 0.1}},
    }
    r = rank(scores)
    assert r[0]["algorithm"] == "a"
