"""Runner: orchestrates a full evaluation run and writes artifacts."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .adapter import ModelAdapter
from .data import Prediction, Sample, _jsonable
from .dataset import Dataset
from .metric import Metric
from .registry import get_adapter, get_dataset, get_metric, get_task
from .reproducibility import collect_env, dumps_jsonable


class Runner:
    """Drives: dataset -> adapter -> predictions -> metrics -> artifacts."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    # ------------------------------------------------------------------ run
    def run(self, output_dir: str) -> Dict[str, Any]:
        """Execute the run described by ``config`` and write all artifacts.

        Returns the metrics dict (also written to ``metrics.json``).
        """
        seed = self.config.get("seed", 0)
        if seed is not None:
            random.seed(seed)
            self._seed_numpy(seed)

        # 1. Resolve components
        task_cls = get_task(self.config["task"])
        dataset = self._build_dataset()
        adapter = self._build_adapter()
        metrics = self._build_metrics(task_cls)

        # 2. Load
        adapter.load()
        samples = dataset.load(self._split)

        # 3. Predict
        predictions = [adapter.predict(s) for s in samples]

        # 4. Metrics
        metrics_out: Dict[str, Any] = {}
        for metric in metrics:
            metrics_out[metric.name] = metric.compute(predictions, samples)

        # 5. Write artifacts
        os.makedirs(output_dir, exist_ok=True)
        self._write_config(output_dir)
        self._write_predictions(output_dir, predictions)
        self._write_metrics(output_dir, metrics_out)
        self._write_env(output_dir, seed, adapter)
        self._write_report(output_dir, metrics_out, len(samples))

        return metrics_out

    # ------------------------------------------------------------- builders
    def _build_dataset(self) -> Dataset:
        ds_cfg = dict(self.config.get("dataset", {}) or {})
        name = ds_cfg.pop("name", "synthetic")
        split = ds_cfg.pop("split", "test")
        cls = get_dataset(name)
        instance = cls(**ds_cfg) if ds_cfg else cls()
        # Store split for load() call below (keep Dataset interface uniform).
        self._split = split
        return instance

    def _build_adapter(self) -> ModelAdapter:
        a_cfg = dict(self.config.get("adapter", {}) or {})
        name = a_cfg.pop("name", None)
        if name is None:
            raise ValueError("config.adapter.name is required")
        cls = get_adapter(name)
        return cls(**a_cfg)

    def _build_metrics(self, task_cls: Any) -> List[Metric]:
        names = self.config.get("metrics") or task_cls.default_metrics()
        metrics: List[Metric] = []
        for name in names:
            mcls = get_metric(name)
            metrics.append(mcls())
        return metrics

    # ------------------------------------------------------------- writers
    def _write_config(self, output_dir: str) -> None:
        import yaml

        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False, allow_unicode=True)

    def _write_predictions(self, output_dir: str, predictions: List[Prediction]) -> None:
        path = os.path.join(output_dir, "predictions.jsonl")
        with open(path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p.to_json(), ensure_ascii=False) + "\n")

    def _write_metrics(self, output_dir: str, metrics: Dict[str, Any]) -> None:
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            f.write(dumps_jsonable(_jsonable(metrics)))

    def _write_env(self, output_dir: str, seed: Optional[int], adapter: ModelAdapter) -> None:
        model_meta = adapter.model_identity() if hasattr(adapter, "model_identity") else {}
        model_meta.setdefault("adapter", adapter.name)
        env = collect_env(seed, model_meta)
        env["run"] = {
            "run_id": os.path.basename(output_dir.rstrip("/")) or "run",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(output_dir, "env.json"), "w") as f:
            f.write(dumps_jsonable(env))

    def _write_report(self, output_dir: str, metrics: Dict[str, Any], n_samples: int) -> None:
        lines = ["# MusicBench Run Report", ""]
        lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"- Samples: {n_samples}")
        lines.append("")
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for group, values in metrics.items():
            if isinstance(values, dict):
                for k, v in _flatten(values).items():
                    lines.append(f"| {group}.{k} | {_fmt(v)} |")
            else:
                lines.append(f"| {group} | {_fmt(values)} |")
        with open(os.path.join(output_dir, "report.md"), "w") as f:
            f.write("\n".join(lines) + "\n")

    # ----------------------------------------------------------------- utils
    @staticmethod
    def _seed_numpy(seed: int) -> None:
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)
