#!/usr/bin/env bash
# musicbench orchestrator: task select -> tier select -> env select ->
# package dataset -> run algorithms (each in its own conda env) ->
# score (weighted ranking) -> plot + log.
#
# Usage:
#   ./run.sh                       # fully interactive
#   ./run.sh text2music            # task by name (still asks tier + env)
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
prompt_choice() { # $1=prompt  $2=default  -> echoes chosen label
  local prompt="$1" default="$2" ans
  printf "%s [%s]: " "$prompt" "$default" >&2
  read -r ans
  echo "${ans:-$default}"
}

# ------------------------------------------------------------------ 1. select task
if [[ -z "$TASK" ]]; then
  echo "Select a task:"
  echo "  1) text2music     (music generation — objective + deep metrics)"
  echo "  2) beat_tracking"
  echo "  3) tagging"
  printf "Choice [1]: "
  read -r choice
  case "${choice:-1}" in
    1|text2music)    TASK=text2music ;;
    2|beat_tracking) TASK=beat_tracking ;;
    3|tagging)       TASK=tagging ;;
    *) die "invalid choice: $choice" ;;
  esac
fi
case "$TASK" in
  text2music|beat_tracking|tagging) ;;
  *) die "unknown task: $TASK" ;;
esac

# ------------------------------------------------------------------ 2. select metric tier
TIER=""
if [[ "$TASK" == "text2music" ]]; then
  echo "Select metric tier:"
  echo "  1) basic        (L0 objective features only, no deep models)"
  echo "  2) professional (L0 + L1: FAD / KL / CLAP — needs deep models)"
  printf "Choice [1]: "
  read -r tier_choice
  case "${tier_choice:-1}" in
    1|basic)        TIER=basic ;;
    2|professional)  TIER=professional ;;
    *) die "invalid tier: $tier_choice" ;;
  esac
else
  TIER=basic
fi
say "task=$TASK  tier=$TIER"

# ------------------------------------------------------------------ 3. harness env (scoring/plotting only)
# Each algorithm declares its OWN env in algorithms/<name>/config.yaml; the
# harness env is separate and only runs package/score/plot (no model weights).
HARNESS_ENV="${BENCH_ENV}"
say "harness env=$HARNESS_ENV (algorithms use their own env from config.yaml)"

# ------------------------------------------------------------------ dirs
OUT="runs/${TASK}_$(date +%Y%m%d_%H%M%S)"
DATASET_DIR="$OUT/dataset"
PRED_DIR="$OUT/predictions"
RESULT_DIR="$OUT/results"
mkdir -p "$DATASET_DIR" "$PRED_DIR" "$RESULT_DIR"

PY="$CONDA run -n $HARNESS_ENV python"

# ------------------------------------------------------------------ 4. package dataset
say "packaging dataset -> $DATASET_DIR"
if [[ "$TASK" == "text2music" ]]; then
  # text2music uses the MusicLite feature JSON (random 10 by default).
  echo "Feature JSON root [/data/MusicLite_data/feature]: "
  read -r feat_root
  feat_root="${feat_root:-/data/MusicLite_data/feature}"
  echo "Number of samples [10]: "
  read -r n_samples
  n_samples="${n_samples:-10}"
  $PY "$HERE/scripts/package_dataset.py" \
    --dataset feature_json --split test \
    --output "$DATASET_DIR" \
    --root "$feat_root" --num-samples "$n_samples" || die "dataset packaging failed"
else
  $PY "$HERE/scripts/package_dataset.py" \
    --dataset synthetic --split test \
    --output "$DATASET_DIR" || die "dataset packaging failed"
fi

# ------------------------------------------------------------------ 5. run each algorithm in ITS OWN env (from config.yaml)
# Discover algorithms for this task.
ALGOS=()
for d in "$HERE"/algorithms/*/; do
  cfg="$d/config.yaml"
  [[ -f "$cfg" ]] || continue
  algo_task=$(grep -E '^\s*task:' "$cfg" | head -1 | awk '{print $2}')
  if [[ "$algo_task" == "$TASK" ]]; then
    ALGOS+=("$(basename "$d")")
  fi
done
[[ ${#ALGOS[@]} -gt 0 ]] || die "no algorithms support task $TASK (add one under algorithms/)"
say "algorithms=[${ALGOS[*]}]"

declare -a LABELS=()
declare -a PRED_FILES=()
for algo in "${ALGOS[@]}"; do
  cfg="$HERE/algorithms/$algo/config.yaml"
  script=$(grep -E '^\s*infer_script:' "$cfg" | head -1 | awk '{print $2}')
  script="${script:-infer.py}"
  # Each algorithm declares its own env; default to harness env if absent.
  algo_env=$(grep -E '^\s*env:' "$cfg" | head -1 | awk '{print $2}')
  algo_env="${algo_env:-$HARNESS_ENV}"

  pred="$PRED_DIR/$algo.jsonl"
  say "running $algo (env=$algo_env) -> $pred"
  "$CONDA" run -n "$algo_env" python "$HERE/algorithms/$algo/$script" \
    --manifest "$DATASET_DIR/manifest.json" --output "$pred" || die "$algo inference failed"
  LABELS+=("$algo")
  PRED_FILES+=("$pred")
done

# ------------------------------------------------------------------ 6. score (weighted ranking)
say "scoring (tier=$TIER)"
$PY "$HERE/scripts/score.py" \
  --manifest "$DATASET_DIR/manifest.json" \
  --task "$TASK" --tier "$TIER" \
  --predictions "${PRED_FILES[@]}" \
  --labels "${LABELS[@]}" \
  --output "$RESULT_DIR" || die "scoring failed"

# ------------------------------------------------------------------ 7. plot + log
say "rendering comparison chart"
$PY "$HERE/scripts/plot.py" \
  --scores "$RESULT_DIR/scores.json" \
  --output "$RESULT_DIR" || warn "plot failed (non-fatal)"

# ------------------------------------------------------------------ done
say "done. artifacts:"
printf '  %s\n' "$DATASET_DIR/manifest.json" "$RESULT_DIR/scores.json" "$RESULT_DIR/report.log" "$RESULT_DIR/comparison.png"
