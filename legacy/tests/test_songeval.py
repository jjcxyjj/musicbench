import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from musicbench.core.data import Prediction, Sample
from musicbench.core.ranking import rank
from musicbench.metrics.generation import _fad
from musicbench.metrics.songeval import SongEvalMetric
from scripts.score import load_predictions, validate_predictions


def test_fad_noncommuting_covariances():
    from scipy.linalg import sqrtm
    rng = np.random.default_rng(42)
    ref = rng.normal(size=(40, 3)) @ np.array([[2, 1, 0], [0, 1, 1], [0, 0, 3]])
    gen = rng.normal(size=(40, 3)) @ np.array([[1, 0, 2], [1, 2, 0], [0, 1, 1]])
    cr, cg = np.cov(ref, rowvar=False), np.cov(gen, rowvar=False)
    diff = ref.mean(0) - gen.mean(0)
    expected = diff @ diff + np.trace(cr + cg - 2 * sqrtm(cr @ cg).real)
    assert _fad(ref, gen) == pytest.approx(expected, abs=1e-8)
    assert _fad(ref, ref) == pytest.approx(0, abs=1e-8)


def test_prediction_contract(tmp_path):
    audio = tmp_path / "a.wav"
    audio.touch()
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"id": "a", "output": {"audio_path": "a.wav"}}))
    preds = load_predictions(str(path))
    assert preds[0].output["audio_path"] == str(audio)
    samples = [Sample(id="a", audio_path=str(audio))]
    validate_predictions("text2music", preds, samples)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_predictions("text2music", preds * 2, samples)
    with pytest.raises(ValueError, match="mismatch"):
        validate_predictions("text2music", [], samples)


def test_ties_and_partial_results():
    scores = {"a": {"songeval": {"musicality": 4, "coverage": 1}},
              "b": {"songeval": {"musicality": 4, "coverage": 1}}}
    assert [r["score"] for r in rank(scores)] == [0.5, 0.5]
    scores["b"]["songeval"]["coverage"] = 0.5
    assert all(not r["used_metrics"] for r in rank(scores))


def test_songeval_worker_bridge(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "ckpt").mkdir(parents=True)
    (repo / "ckpt/model.safetensors").touch()
    monkeypatch.setenv("MUSICBENCH_SONGEVAL_REPO", str(repo))
    monkeypatch.setenv("MUSICBENCH_SONGEVAL_PYTHON", "/isolated/python")

    def fake_run(command, **kwargs):
        assert command[0] == "/isolated/python"
        rows = json.loads(Path(command[command.index("--input") + 1]).read_text())
        assert rows[0]["has_vocal"] is False
        output = {"per_sample": [{"id": "a", "coherence": 4.0, "musicality": 3.0,
                                  "memorability": 2.0, "clarity": 4.5, "naturalness": None}],
                  "errors": [], "model": {"commit": "test"}}
        Path(command[command.index("--output") + 1]).write_text(json.dumps(output))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("musicbench.metrics.songeval.subprocess.run", fake_run)
    out = SongEvalMetric().compute([Prediction(id="a", output={"audio_path": "a.wav"})],
                                  [Sample(id="a", audio_path="", metadata={"has_vocal": False})])
    assert out["coverage"] == 1
    assert out["musicality"] == 3
    assert out["naturalness"] is None


def test_songeval_missing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSICBENCH_SONGEVAL_REPO", str(tmp_path))
    assert SongEvalMetric().compute([], [])["available"] is False


def test_kl_infers_each_clip_separately(monkeypatch):
    import sys
    from musicbench.metrics.generation import KLDivergenceMetric
    calls = []

    class AudioTagging:
        def __init__(self, **kwargs):
            pass

        def inference(self, audio):
            calls.append(audio.shape)
            return np.array([[0.2, 0.8]]), None

    monkeypatch.setitem(sys.modules, "panns_inference", SimpleNamespace(AudioTagging=AudioTagging))
    monkeypatch.setattr("musicbench.metrics.generation._generated_paths", lambda _: ["a", "b"])
    monkeypatch.setattr("musicbench.metrics.generation._reference_paths", lambda _: ["c", "d"])
    monkeypatch.setattr("musicbench.metrics.generation._load", lambda _: np.ones(120))
    result = KLDivergenceMetric().compute([], [])
    assert calls == [(1, 120)] * 4
    assert result["kl"] == pytest.approx(0)


def test_plot_without_ranking(tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.plot import _plot
    _plot(str(tmp_path), {"a": {"clap_score.clap": 0.3}}, ["clap_score.clap"], "text2music", [])
    assert (tmp_path / "comparison.png").is_file()
