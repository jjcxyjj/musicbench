# musicbench

A **music algorithm evaluation harness**. v0.1 ships two tasks — **beat
tracking** and **tagging** — with a registry, a YAML-driven runner, and a CLI
(`run` / `evaluate` / `report`). The abstraction is designed to grow into
chord, source separation, QA, and text-to-music tasks without rewriting the core.

## Layout

```
musicbench/
├── core/          Sample, Prediction, ModelAdapter, Metric, Task, Dataset,
│                  Runner, Registry, reproducibility
├── tasks/         beat_tracking, tagging
├── metrics/       mir_eval.beat F-measure, tempo MAE, mAP/F1/AUC (sklearn)
├── datasets/      synthetic (deterministic, no download)
├── adapters/      dsp_beat, dsp_tag (dependency-light demo adapters)
├── scripts/       package_dataset.py, score.py, plot.py (orchestration)
├── algorithms/    dsp_beat/, dsp_tag/ (each with its own conda env + infer.py)
envs/              benchmark.yml (harness env), algorithm.template.yml
configs/           mvp.yaml (beat), tagging.yaml
examples/          beat_model.py, tag_model.py (adapter contract examples)
tests/             pytest suite (mir_eval alignment + end-to-end)
run.sh             one-command orchestrator (task select -> infer -> score -> png/log)
```

## Two-layer environment design

Different algorithms need conflicting dependencies, so the harness and each
algorithm run in **separate conda environments**, bridged only by JSON files:

```
benchmark env (minimal, no torch)  <-- scores, plots, orchestrates
        ▲ reads predictions.jsonl
   ┌────┴───────┬───────────┬─────────┐
   ▼            ▼           ▼         ▼
alg A env     alg B env  alg C env  ...
  └── each writes predictions.jsonl (pure JSON, no musicbench import)
```

* `envs/benchmark.yml` — the harness env (numpy/scipy/sklearn/mir_eval/matplotlib).
* `envs/algorithm.template.yml` — copy per algorithm and add its real deps.
* `algorithms/<name>/config.yaml` declares the algorithm's `env` and `task`.

## Install (harness env)

```bash
conda env create -f envs/benchmark.yml
conda activate musicbench
pip install -e .            # installs the `musicbench` CLI
```

## One-command orchestration

```bash
./run.sh                 # interactive: pick task, runs everything
./run.sh beat_tracking   # non-interactive
./run.sh tagging
```

This packages the dataset, runs each matching algorithm in its own conda env,
scores them, and writes a comparison PNG + log:

```
runs/<task>_<timestamp>/
├── dataset/manifest.json          # inputs + ground truth
├── predictions/<algo>.jsonl       # one file per algorithm
└── results/
    ├── scores.json                # per-algorithm metrics
    ├── report.log                 # human-readable ranking
    └── comparison.png             # bar chart
```

## Manual step-by-step

```bash
# 1. package dataset
python scripts/package_dataset.py --dataset synthetic --output dataset/

# 2. run an algorithm (its own env)
conda run -n <algo_env> python algorithms/dsp_beat/infer.py \
    --manifest dataset/manifest.json --output preds/dsp_beat.jsonl

# 3. score
python scripts/score.py --manifest dataset/manifest.json --task beat_tracking \
    --predictions preds/dsp_beat.jsonl --labels dsp_beat --output results/

# 4. plot + log
python scripts/plot.py --scores results/scores.json --output results/
```

## Adding a new algorithm

1. Copy `envs/algorithm.template.yml` to `envs/<algo>.yml`, fill in deps, `conda env create -f envs/<algo>.yml`.
2. Create `algorithms/<algo>/` with `config.yaml` (`name`, `env`, `task`) and `infer.py` (reads `manifest.json`, writes `predictions.jsonl`). `infer.py` must NOT import musicbench.
3. `./run.sh <task>` will auto-discover it.

## Run

```bash
# Full run: dataset -> adapter -> predictions -> metrics -> artifacts
musicbench run --config configs/mvp.yaml --output runs/mvp

# Offline evaluate an existing predictions.jsonl
musicbench evaluate --task beat_tracking --predictions runs/mvp/predictions.jsonl \
    --dataset synthetic --split test

# Render a markdown report from a run directory
musicbench report --run runs/mvp --format markdown
```

A run directory contains:

```
runs/mvp/
├── config.yaml          # the config used (reproducible)
├── predictions.jsonl    # one JSON line per sample: {id, output}
├── metrics.json         # metric name -> values
├── report.md            # human-readable summary
└── env.json             # seed, python/torch/cuda versions, code_commit, model hash
```

## Tests

```bash
pytest -q
```

## Writing your own model adapter

Subclass `ModelAdapter`, implement `load()` and `predict(sample)`, and register
it so a YAML config can reference it:

```python
from musicbench.core.adapter import ModelAdapter
from musicbench.core.data import Prediction, Sample
from musicbench.core.registry import ADAPTERS

@ADAPTERS.register("my_beat_model")
class MyBeatModel(ModelAdapter):
    name = "my_beat_model"
    def load(self): ...
    def predict(self, sample: Sample) -> Prediction:
        return Prediction(id=sample.id, output={"beats": [...], "tempo": 120.0})
```

Then point `adapter.name` in your YAML at `my_beat_model`. See
`examples/beat_model.py` and `examples/tag_model.py`.
