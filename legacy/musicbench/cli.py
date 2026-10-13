"""Command-line interface: run / evaluate / report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# Import built-in registrations (side effects) so registry names resolve.
import musicbench.tasks  # noqa: F401
import musicbench.metrics  # noqa: F401
import musicbench.datasets  # noqa: F401
import musicbench.adapters  # noqa: F401


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="musicbench", description="Music algorithm evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a full evaluation from a YAML config")
    p_run.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/mvp.yaml)")
    p_run.add_argument("--output", required=True, help="Run output directory (e.g. runs/mvp)")

    p_eval = sub.add_parser("evaluate", help="Evaluate an existing predictions.jsonl")
    p_eval.add_argument("--task", required=True, help="Task name (beat_tracking | tagging)")
    p_eval.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    p_eval.add_argument("--dataset", default="synthetic", help="Dataset registry name")
    p_eval.add_argument("--split", default="test", help="Dataset split")
    p_eval.add_argument("--metrics", nargs="*", help="Override metric names")
    p_eval.add_argument("--output", help="Optional JSON path to write metrics")

    p_rep = sub.add_parser("report", help="Render a run directory to markdown")
    p_rep.add_argument("--run", required=True, help="Run directory (e.g. runs/mvp)")
    p_rep.add_argument("--format", default="markdown", choices=["markdown"], help="Report format")
    p_rep.add_argument("--output", help="Report output path (default: <run>/report.md)")

    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "report":
        return _cmd_report(args)
    parser.error("unknown command")
    return 2


# --------------------------------------------------------------------- run
def _cmd_run(args: argparse.Namespace) -> int:
    import yaml

    from musicbench.core.runner import Runner

    with open(args.config) as f:
        config = yaml.safe_load(f)

    runner = Runner(config)
    metrics = runner.run(args.output)
    print(f"[run] wrote artifacts to {args.output}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


# ----------------------------------------------------------------- evaluate
def _cmd_evaluate(args: argparse.Namespace) -> int:
    from musicbench.core.data import prediction_from_json
    from musicbench.core.registry import get_dataset, get_metric, get_task

    # Load predictions.
    predictions = []
    with open(args.predictions) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(prediction_from_json(json.loads(line)))

    # Load dataset ground truth.
    ds_cfg = {"name": args.dataset, "split": args.split}
    ds_cls = get_dataset(args.dataset)
    samples = ds_cls().load(args.split)

    # Resolve metrics.
    task_cls = get_task(args.task)
    metric_names = args.metrics or task_cls.default_metrics()

    out: Dict[str, Any] = {}
    for name in metric_names:
        m = get_metric(name)()
        out[name] = m.compute(predictions, samples)

    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return 0


# ------------------------------------------------------------------ report
def _cmd_report(args: argparse.Namespace) -> int:
    from musicbench.core.runner import _flatten, _fmt

    run_dir = args.run
    metrics_path = os.path.join(run_dir, "metrics.json")
    config_path = os.path.join(run_dir, "config.yaml")
    env_path = os.path.join(run_dir, "env.json")

    with open(metrics_path) as f:
        metrics = json.load(f)

    lines = ["# MusicBench Run Report", ""]
    for meta_path, title in [(env_path, "Environment"), (config_path, "Config")]:
        if os.path.exists(meta_path):
            lines.append(f"## {title}")
            lines.append("")
            with open(meta_path) as f:
                if meta_path.endswith(".json"):
                    meta = json.load(f)
                    for k, v in _flatten(meta).items():
                        lines.append(f"- **{k}**: {v}")
                else:
                    lines.append("```yaml")
                    lines.append(f.read().strip())
                    lines.append("```")
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
    lines.append("")

    report = "\n".join(lines) + "\n"
    output = args.output or os.path.join(run_dir, "report.md")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write(report)
    print(f"[report] wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
