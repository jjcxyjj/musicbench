"""Score algorithm predictions against a packaged dataset.

Reads ``manifest.json`` (ground truth) + one or more ``predictions.jsonl``
files and writes a combined ``scores.json``, then optionally a comparison PNG
and a log file.

Usage:
    python scripts/score.py --manifest dataset/manifest.json \
        --task beat_tracking \
        --predictions preds/algA.jsonl preds/algB.jsonl \
        --labels algA algB --output results/
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import musicbench.tasks  # noqa: F401
import musicbench.metrics  # noqa: F401
import musicbench.datasets  # noqa: F401

from musicbench.core.data import Sample, prediction_from_json
from musicbench.core.ranking import rank
from musicbench.core.registry import get_metric, get_task
from musicbench.core.runner import _flatten
from musicbench.core.reproducibility import collect_env, dumps_jsonable


def load_manifest(path: str) -> List[Sample]:
    with open(path) as f:
        rows = json.load(f)
    samples: List[Sample] = []
    for r in rows:
        base = os.path.dirname(path)
        audio_path = r["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(base, audio_path)
        samples.append(
            Sample(
                id=r["id"],
                audio_path=audio_path,
                metadata=r.get("metadata", {}),
                ground_truth=r.get("ground_truth", {}),
            )
        )
    return samples


def load_predictions(path: str) -> List:
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pred = prediction_from_json(json.loads(line))
                if isinstance(pred.output, dict) and pred.output.get("audio_path"):
                    audio = pred.output["audio_path"]
                    if not os.path.isabs(audio):
                        pred.output["audio_path"] = os.path.abspath(os.path.join(os.path.dirname(path), audio))
                preds.append(pred)
    return preds


def validate_predictions(task_name, predictions, samples):
    expected = [s.id for s in samples]
    actual = [p.id for p in predictions]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Manifest is empty or contains duplicate IDs")
    if len(set(actual)) != len(actual):
        raise ValueError("Duplicate prediction IDs")
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise ValueError(f"Prediction IDs mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    if task_name == "text2music":
        for pred in predictions:
            path = pred.output.get("audio_path") if isinstance(pred.output, dict) else None
            if not path or not os.path.isfile(path):
                raise ValueError(f"Missing generated audio for {pred.id}: {path}")


def score_one(task_name: str, metric_names: List[str], predictions: List, samples: List[Sample]) -> Dict[str, Any]:
    validate_predictions(task_name, predictions, samples)
    out: Dict[str, Any] = {}
    for name in metric_names:
        m = get_metric(name)()
        out[name] = m.compute(predictions, samples)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Score predictions against a packaged dataset")
    p.add_argument("--manifest", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", help="Algorithm names, aligned with --predictions")
    p.add_argument("--output", required=True)
    p.add_argument("--metrics", nargs="+", help="Explicit metric names, overriding tier defaults")
    p.add_argument("--require-metrics", action="store_true", help="Fail on unavailable or incomplete metrics")
    p.add_argument("--tier", default="basic", choices=["basic", "professional"],
                   help="Metric tier: basic (no deep models) or professional (all)")
    args = p.parse_args()

    samples = load_manifest(args.manifest)
    task_cls = get_task(args.task)
    metric_names = args.metrics or task_cls.default_metrics(args.tier)

    if args.labels and len(args.labels) != len(args.predictions):
        raise SystemExit("--labels must match --predictions in length")
    labels = args.labels or [f"alg{i}" for i in range(len(args.predictions))]
    if len(set(labels)) != len(labels):
        raise SystemExit("Algorithm labels must be unique")

    os.makedirs(args.output, exist_ok=True)

    all_scores: Dict[str, Dict[str, Any]] = {}
    for label, pred_path in zip(labels, args.predictions):
        preds = load_predictions(pred_path)
        scores = score_one(args.task, metric_names, preds, samples)
        if args.require_metrics:
            for name, values in scores.items():
                if values.get("available") is False or values.get("coverage", 1.0) < 1.0:
                    raise RuntimeError(f"{label}: required metric {name} unavailable/incomplete: {values.get('reason', values.get('errors'))}")
        all_scores[label] = scores

    # Aggregate into a flat, comparison-friendly table.
    flat: Dict[str, Dict[str, float]] = {}
    for label, scores in all_scores.items():
        flat[label] = {k: v for k, v in _flatten(scores).items() if isinstance(v, (int, float))}

    ranking = rank(all_scores)

    with open(os.path.join(args.output, "scores.json"), "w") as f:
        f.write(dumps_jsonable({"task": args.task, "tier": args.tier,
                                "metrics": metric_names,
                                "ranking_note": "Relative min-max ranking within this comparison, not absolute quality. Basic audio features measure reference similarity, not aesthetics.",
                                "algorithms": all_scores, "ranking": ranking}))

    with open(os.path.join(args.output, "env.json"), "w") as f:
        f.write(dumps_jsonable(collect_env(None)))

    print(json.dumps(all_scores, indent=2, ensure_ascii=False))
    print("\n# Ranking (weighted mix)")
    for i, r in enumerate(ranking, 1):
        print(f"  {i}. {r['algorithm']}: {r['score']:.4f}  (used: {len(r['used_metrics'])}, skipped: {len(r['skipped_metrics'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
