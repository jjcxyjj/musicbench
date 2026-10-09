# SongEval integration

Upstream: https://github.com/ASLP-lab/SongEval

The downloaded checkout lives in `third_party/SongEval` (ignored by Git).
The integration runs upstream `Generator` and its checkpoint with MuQ hidden
state 6, mono 24 kHz audio, eval mode and inference mode, matching `eval.py`.
Weights are loaded strictly. No upstream source is modified or redistributed.

## Setup

```bash
git clone https://github.com/ASLP-lab/SongEval.git third_party/SongEval
conda env create -f envs/songeval.yml
export MUSICBENCH_SONGEVAL_PYTHON="$(conda run -n musicbench-songeval python -c 'import sys; print(sys.executable)')"
export MUSICBENCH_SONGEVAL_DEVICE=cpu  # auto uses CUDA if available, otherwise CPU
```

First execution downloads `OpenMuQ/MuQ-large-msd-iter`; allow network access
or populate the Hugging Face cache first. The scoring checkpoint is provided
by the upstream repository. CPU full-song inference can be slow and memory
intensive. Audio is not silently truncated or chunked, since that changes the
meaning of full-song structure and coherence scores.

## Run

`./run.sh text2music`, professional tier, includes SongEval. Alternatively:

```bash
python scripts/score.py --manifest dataset/manifest.json --task text2music \
  --predictions preds/base.jsonl preds/lora.jsonl --labels base lora \
  --metrics songeval --require-metrics --output runs/songeval
python scripts/plot.py --scores runs/songeval/scores.json --output runs/songeval
```

Settings: `MUSICBENCH_SONGEVAL_REPO` overrides the checkout location;
`MUSICBENCH_SONGEVAL_TIMEOUT` sets the worker timeout in seconds (default 3600).
The Python variable must be an executable path, not a shell command.

## Interpretation

Coherence, musicality, memorability and clarity are averaged separately,
with raw per-sample scores, errors, coverage, checkpoint SHA256, upstream Git
commit and runtime device recorded in scores.json. Outputs are on a 1-5 scale.
Naturalness concerns vocal breathing and phrasing: it is reported only for
samples with JSON boolean `metadata.has_vocal: true`, and excluded from the
default ranking. Unknown vocal status and instrumentals receive null.
Incomplete SongEval results do not enter ranking. `--require-metrics` makes
unavailable/incomplete metrics fatal; the default reports them as unavailable.

SongEval measures predicted aesthetics, not erhu/pipa identity or prompt
adherence. For Chinese instrumental LoRA evaluation, retain expert blind
listening and instrument-specific checks. Domain validity on this material
has not been established by this integration.

Rank scores are relative to the compared algorithms and weights. Basic
feature MAE measures similarity to a reference, not musical beauty; prefer
`--metrics songeval clap_score` for a focused aesthetics/alignment comparison.
Use enough independent songs, fixed prompts/seeds and separate held-out
performers/compositions; ten clips are only a smoke test.

## Upstream licensing

Upstream README declares CC BY-NC-SA 4.0 / non-commercial use, while its
LICENSE file contains Apache-2.0. This inconsistency remains unresolved;
confirm applicability with the authors before commercial use of code/weights.
Keep upstream attribution and cite Yao et al., SongEval (arXiv:2505.10793).

## Other benchmark fixes

- FAD uses the symmetric covariance sandwich square root; both distributions
  require at least two samples. Embeddings here are CLAP, so disclose that
  variant when comparing published FAD values.
- KL computes per-clip PANNs outputs then normalizes mean class scores into
  distributions. It is a dataset-level normalized-tag KL, not paired KL.
- Scoring rejects missing/duplicate/unknown IDs, duplicate labels and missing
  generated files. Relative generated paths resolve against predictions.jsonl.
- Tied metrics receive equal neutral ranking scores. Unavailable and partial
  metric groups are excluded consistently across algorithms.
- Plots exclude counters/booleans and show missing values as missing.
