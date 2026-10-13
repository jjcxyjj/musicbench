"""Isolated metric job: JSON in/out, no orchestration dependencies required."""
import argparse
import json
from pathlib import Path
import sys
import traceback

# Permit running directly in an existing evaluation environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def clean(value):
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    job = json.loads(Path(args.request).read_text())
    try:
        from musicbench.evaluation.primary import compute
        result = compute(job)
    except Exception as exc:
        traceback.print_exc()
        result = {'available': False, 'coverage': 0.0, 'reason': f'{type(exc).__name__}: {exc}'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
