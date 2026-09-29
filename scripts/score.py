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
                preds.append(prediction_from_json(json.loads(line)))
    return preds


def score_one(task_name: str, metric_names: List[str], predictions: List, samples: List[Sample]) -> Dict[str, Any]:
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
    p.add_argument("--tier", default="basic", choices=["basic", "professional"],
                   help="Metric tier: basic (no deep models) or professional (all)")
    args = p.parse_args()

    samples = load_manifest(args.manifest)
    task_cls = get_task(args.task)
    metric_names = task_cls.default_metrics(args.tier)

    if args.labels and len(args.labels) != len(args.predictions):
        raise SystemExit("--labels must match --predictions in length")
    labels = args.labels or [f"alg{i}" for i in range(len(args.predictions))]

    os.makedirs(args.output, exist_ok=True)

    all_scores: Dict[str, Dict[str, Any]] = {}
    for label, pred_path in zip(labels, args.predictions):
        preds = load_predictions(pred_path)
        scores = score_one(args.task, metric_names, preds, samples)
        all_scores[label] = scores

    # Aggregate into a flat, comparison-friendly table.
    flat: Dict[str, Dict[str, float]] = {}
    for label, scores in all_scores.items():
        flat[label] = {k: v for k, v in _flatten(scores).items() if isinstance(v, (int, float))}

    ranking = rank(all_scores)

    with open(os.path.join(args.output, "scores.json"), "w") as f:
        f.write(dumps_jsonable({"task": args.task, "tier": args.tier,
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
