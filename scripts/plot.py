"""Render a comparison bar chart (PNG) + log file from scores.json.

Usage:
    python scripts/plot.py --scores results/scores.json --output results/
"""

from __future__ import annotations

import argparse
import json
import os
import math
from typing import Any, Dict, List

from musicbench.core.runner import _flatten


def main() -> int:
    p = argparse.ArgumentParser(description="Plot algorithm comparison from scores.json")
    p.add_argument("--scores", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.scores) as f:
        data = json.load(f)

    algorithms: Dict[str, Dict[str, Any]] = data.get("algorithms", {})
    task = data.get("task", "unknown")
    ranking = data.get("ranking", [])

    # Flatten each algorithm's metrics into scalar key -> value.
    rows: Dict[str, Dict[str, float]] = {}
    metric_keys: List[str] = []
    for label, scores in algorithms.items():
        from musicbench.core.ranking import DEFAULT_WEIGHTS
        flat = {k: v for k, v in _flatten(scores).items()
                if k in DEFAULT_WEIGHTS and isinstance(v, (int, float))
                and not isinstance(v, bool) and math.isfinite(v)}
        rows[label] = flat
        for k in flat:
            if k not in metric_keys:
                metric_keys.append(k)

    # Write a human-readable log (ranking first).
    log_lines = [f"# musicbench report — task: {task}", ""]
    log_lines.append(data.get("ranking_note", ""))
    for label, groups in algorithms.items():
        for name, values in groups.items():
            if isinstance(values, dict) and (values.get("available") is False or values.get("coverage", 1.0) < 1.0):
                log_lines.append(f"WARNING {label}/{name}: {values.get('reason', values.get('errors', 'incomplete coverage'))}")
    if ranking:
        log_lines.append("## Ranking (weighted mix)")
        for i, r in enumerate(ranking, 1):
            log_lines.append(f"  {i}. {r['algorithm']}: {r['score']:.4f}")
        log_lines.append("")
    for label in rows:
        log_lines.append(f"## {label}")
        for k in metric_keys:
            if k in rows[label]:
                log_lines.append(f"  {k}: {rows[label][k]:.6f}")
        log_lines.append("")
    with open(os.path.join(args.output, "report.log"), "w") as f:
        f.write("\n".join(log_lines))

    # Render PNG bar chart (raw metrics + ranking side panel).
    try:
        _plot(args.output, rows, metric_keys, task, ranking)
    except Exception as exc:  # pragma: no cover - matplotlib may be absent
        print(f"[plot] PNG render skipped ({exc})")
        return 0

    print(f"[plot] wrote report.log and comparison.png to {args.output}")
    return 0


def _plot(output_dir: str, rows: Dict[str, Dict[str, float]], metric_keys: List[str],
          task: str, ranking: List[Dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(rows.keys())
    x = list(range(len(metric_keys)))
    width = 0.8 / max(1, len(labels))

    has_ranking = bool(ranking)
    if has_ranking:
        fig, (ax, ax_rank) = plt.subplots(1, 2, figsize=(max(8, len(metric_keys) * 1.8), 4.8),
                                         gridspec_kw={"width_ratios": [3, 1]})
    else:
        fig, ax = plt.subplots(1, 1, figsize=(max(8, len(metric_keys) * 1.8), 4.8))

    for i, label in enumerate(labels):
        vals = [rows[label].get(k, float("nan")) for k in metric_keys]
        ax.bar([xi + (i - len(labels) / 2 + 0.5) * width for xi in x], vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_title(f"musicbench — {task}")
    ax.legend()

    if has_ranking:
        names = [r["algorithm"] for r in reversed(ranking)]
        scores = [r["score"] for r in reversed(ranking)]
        ax_rank.barh(names, scores)
        ax_rank.set_xlim(0, 1)
        ax_rank.set_title("ranking (weighted)")
        ax_rank.set_xlabel("score")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
