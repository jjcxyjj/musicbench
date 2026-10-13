#!/usr/bin/env bash
# Launch orchestration in the harness environment; each worker has its own env.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${BENCH_ENV:-musicbench}" == "current" ]]; then
  exec "${PYTHON:-python}" -m musicbench.workflow "$@"
fi
exec "${CONDA:-conda}" run --no-capture-output -n "${BENCH_ENV:-musicbench}" python -m musicbench.workflow "$@"
