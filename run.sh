#!/usr/bin/env bash
# musicbench orchestrator: package dataset -> run algorithms (each in its own
# conda env) -> score -> plot + log.
#
# Usage:
#   ./run.sh                 # interactive task selection
#   ./run.sh beat_tracking   # non-interactive
#   ./run.sh tagging
#
# Environment variables:
#   BENCH_ENV    conda env for the harness itself (default: musicbench)
#   CONDA        path to conda (default: auto-detect)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ENV="${BENCH_ENV:-musicbench}"
CONDA="${CONDA:-$(command -v conda || echo conda)}"

TASK="${1:-}"

# ------------------------------------------------------------------ helpers
say()  { printf '\033[1;32m[run]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ select task
if [[ -z "$TASK" ]]; then
  echo "Select a task:"
  echo "  1) beat_tracking"
  echo "  2) tagging"
  printf "Choice [1]: "
  read -r choice
  case "${choice:-1}" in
    1|beat_tracking) TASK=beat_tracking ;;
    2|tagging)       TASK=tagging ;;
    *) die "invalid choice: $choice" ;;
  esac
fi
[[ "$TASK" == "beat_tracking" || "$TASK" == "tagging" ]] || die "unknown task: $TASK"

# Discover algorithms that support this task.
ALGOS=()
for d in "$HERE"/algorithms/*/; do
  cfg="$d/config.yaml"
  [[ -f "$cfg" ]] || continue
  # crude yaml parse: look for `task:` line
  algo_task=$(grep -E '^\s*task:' "$cfg" | head -1 | awk '{print $2}')
  if [[ "$algo_task" == "$TASK" ]]; then
    ALGOS+=("$(basename "$d")")
  fi
done
[[ ${#ALGOS[@]} -gt 0 ]] || die "no algorithms support task $TASK"

say "task=$TASK  algorithms=[${ALGOS[*]}]"

# ------------------------------------------------------------------ dirs
OUT="runs/${TASK}_$(date +%Y%m%d_%H%M%S)"
DATASET_DIR="$OUT/dataset"
PRED_DIR="$OUT/predictions"
RESULT_DIR="$OUT/results"
mkdir -p "$DATASET_DIR" "$PRED_DIR" "$RESULT_DIR"

# ------------------------------------------------------------------ 1. package dataset
say "packaging dataset -> $DATASET_DIR"
PY="$CONDA run -n $BENCH_ENV python"
$PY "$HERE/scripts/package_dataset.py" \
  --dataset synthetic --split test \
  --output "$DATASET_DIR" || die "dataset packaging failed"

# ------------------------------------------------------------------ 2. run each algorithm in its env
declare -a LABELS=()
declare -a PRED_FILES=()
for algo in "${ALGOS[@]}"; do
  cfg="$HERE/algorithms/$algo/config.yaml"
  env_name=$(grep -E '^\s*env:' "$cfg" | head -1 | awk '{print $2}')
  script=$(grep -E '^\s*infer_script:' "$cfg" | head -1 | awk '{print $2}')
  env_name="${env_name:-$BENCH_ENV}"
  script="${script:-infer.py}"

  pred="$PRED_DIR/$algo.jsonl"
  say "running $algo (env=$env_name) -> $pred"
  "$CONDA" run -n "$env_name" python "$HERE/algorithms/$algo/$script" \
    --manifest "$DATASET_DIR/manifest.json" --output "$pred" || die "$algo inference failed"

  LABELS+=("$algo")
  PRED_FILES+=("$pred")
done

# ------------------------------------------------------------------ 3. score
say "scoring predictions"
$PY "$HERE/scripts/score.py" \
  --manifest "$DATASET_DIR/manifest.json" \
  --task "$TASK" \
  --predictions "${PRED_FILES[@]}" \
  --labels "${LABELS[@]}" \
  --output "$RESULT_DIR" || die "scoring failed"

# ------------------------------------------------------------------ 4. plot + log
say "rendering comparison chart"
$PY "$HERE/scripts/plot.py" \
  --scores "$RESULT_DIR/scores.json" \
  --output "$RESULT_DIR" || warn "plot failed (non-fatal)"

# ------------------------------------------------------------------ done
say "done. artifacts:"
printf '  %s\n' "$DATASET_DIR/manifest.json" "$RESULT_DIR/scores.json" "$RESULT_DIR/report.log" "$RESULT_DIR/comparison.png"
