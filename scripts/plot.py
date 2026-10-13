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

    if data.get("metric_definitions"):
        from musicbench.evaluation.report import render
        render(data, args.output)
        print(f"[report] wrote category charts and analysis to {args.output}")
        return 0

    algorithms: Dict[str, Dict[str, Any]] = data.get("algorithms", {})
    task = data.get("task", "unknown")
    ranking = data.get("ranking", [])

    os.makedirs(args.output, exist_ok=True)
    # Flatten each algorithm's metrics into scalar key -> value.
    rows: Dict[str, Dict[str, float]] = {}
    metric_keys: List[str] = []
    for label, scores in algorithms.items():
        from musicbench.core.ranking import DEFAULT_WEIGHTS
        flat = {k: v for k, v in _flatten(scores).items()
                if k in data.get("ranking_weights", DEFAULT_WEIGHTS) and isinstance(v, (int, float))
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

    log_lines.extend(["## Interpretation", "",
                      f"Samples: {data.get('sample_count', 'unknown')}; seed: {data.get('seed', 'unknown')}.",
                      "All algorithms use the same sampled manifest. Scores do not establish statistical significance."])
    from musicbench.core.ranking import DEFAULT_WEIGHTS
    weights = data.get("ranking_weights", DEFAULT_WEIGHTS)
    for key in metric_keys:
        values = {label: row[key] for label, row in rows.items() if key in row}
        higher = weights[key][1]
        best = (max if higher else min)(values.values())
        winners = [label for label, value in values.items() if value == best]
        log_lines.append(f"- {key}: {'higher' if higher else 'lower'} is better; "
                         f"best observed: {', '.join(winners)} ({best:.6f}); "
                         f"available for {len(values)}/{len(rows)} algorithms.")
    if ranking:
        used = ranking[0]['used_metrics']
        if not used:
            log_lines.append("No shared complete metrics: no valid aggregate ranking is available.")
        else:
            best = ranking[0]['score']
            leaders = [r['algorithm'] for r in ranking if r['score'] == best]
            log_lines.append(f"Aggregate leaders/ties: {', '.join(leaders)}; based on {len(used)} shared complete metrics.")
        skipped = ranking[0].get('skipped_metrics', [])
        if skipped:
            log_lines.append("Excluded from ranking: " + ", ".join(skipped))
    log_lines.extend(["", "SongEval naturalness concerns vocal breathing/phrasing and is omitted for non-vocal or unknown-vocal samples.",
                      "Reference similarity alone does not demonstrate generation quality; copying references can score well.",
                      "", "![Metric comparison](comparison.png)"])
    for filename in ('report.log', 'report.md'):
        with open(os.path.join(args.output, filename), 'w') as handle:
            handle.write("\n".join(log_lines) + "\n")

    # Render PNG bar chart (raw metrics + ranking side panel).
    try:
        _plot(args.output, rows, metric_keys, task, ranking)
    except Exception as exc:  # pragma: no cover - matplotlib may be absent
        print(f"[plot] PNG render skipped ({exc})")
        return 1

    print(f"[plot] wrote report.log and comparison.png to {args.output}")
    return 0


def _plot(output_dir: str, rows: Dict[str, Dict[str, float]], metric_keys: List[str],
          task: str, ranking: List[Dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(rows.keys())
    panels = max(1, len(metric_keys)) + bool(ranking and ranking[0].get('used_metrics'))
    cols = min(3, panels)
    nrows = (panels + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 3.8 * nrows), squeeze=False)
    axes = list(axes.flat)
    for ax, key in zip(axes, metric_keys):
        ax.bar(labels, [rows[label].get(key, float('nan')) for label in labels])
        ax.set_title(key, fontsize=9)
        ax.tick_params(axis='x', rotation=25)
    used = len(metric_keys)
    if not metric_keys:
        axes[0].text(0.5, 0.5, 'No available comparable scalar metrics', ha='center')
        used = 1
    if ranking and ranking[0].get('used_metrics'):
        axes[used].barh([r['algorithm'] for r in reversed(ranking)], [r['score'] for r in reversed(ranking)])
        axes[used].set_xlim(0, 1)
        axes[used].set_title('Relative weighted comparison')
        used += 1
    for ax in axes[used:]:
        ax.set_visible(False)
    fig.suptitle(f'musicbench — {task}')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
