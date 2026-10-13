"""Runner tests: full run produces all artifacts + end-to-end flow."""

import json
import os

import pytest

import musicbench.tasks  # noqa: F401
import musicbench.metrics  # noqa: F401
import musicbench.datasets  # noqa: F401
import musicbench.adapters  # noqa: F401

from musicbench.core.runner import Runner


def _config():
    return {
        "seed": 42,
        "task": "beat_tracking",
        "dataset": {"name": "synthetic", "split": "test", "num_samples": 4, "duration": 6.0},
        "adapter": {"name": "dsp_beat"},
        "metrics": ["beat_fmeasure", "tempo_mae"],
    }


def test_run_writes_all_artifacts(tmp_path):
    out = str(tmp_path / "runs" / "mvp")
    metrics = Runner(_config()).run(out)

    for fname in ["config.yaml", "predictions.jsonl", "metrics.json", "report.md", "env.json"]:
        assert os.path.isfile(os.path.join(out, fname)), f"missing {fname}"

    # metrics.json contains the two metric groups
    with open(os.path.join(out, "metrics.json")) as f:
        m = json.load(f)
    assert "beat_fmeasure" in m
    assert "tempo_mae" in m

    # predictions.jsonl has one line per sample
    with open(os.path.join(out, "predictions.jsonl")) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 4

    # env.json records reproducibility fields
    with open(os.path.join(out, "env.json")) as f:
        env = json.load(f)
    assert env["seed"] == 42
    assert "python" in env
    assert "code_commit" in env
    assert "model" in env


def test_runner_unknown_metric_raises():
    cfg = _config()
    cfg["metrics"] = ["does_not_exist"]
    with pytest.raises(KeyError):
        Runner(cfg).run(str(__import__("tempfile").mkdtemp()))


def test_runner_unknown_adapter_raises():
    cfg = _config()
    cfg["adapter"]["name"] = "no_such_adapter"
    with pytest.raises(KeyError):
        Runner(cfg).run(str(__import__("tempfile").mkdtemp()))
