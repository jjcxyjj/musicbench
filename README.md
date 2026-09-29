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
configs/           mvp.yaml (beat), tagging.yaml
examples/          beat_model.py, tag_model.py (adapter contract examples)
tests/             pytest suite (mir_eval alignment + end-to-end)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs deps + `musicbench` CLI
pip install -e ".[dev]"     # + pytest
```

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
