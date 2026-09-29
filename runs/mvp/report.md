# MusicBench Run Report

## Environment

- **code_commit**: None
- **cuda.available**: False
- **cuda.device_name**: None
- **cuda.version**: None
- **mir_eval_version**: 0.8.2
- **model.adapter**: dsp_beat
- **numpy_version**: 2.4.6
- **python.executable**: /Users/jiaoyuanxiang/Desktop/huawei/.musicbench-venv/bin/python
- **python.machine**: x86_64
- **python.platform**: macOS-10.16-x86_64-i386-64bit
- **python.version**: 3.11.7
- **run.run_id**: mvp
- **run.timestamp**: 2026-09-29T02:35:08.401582+00:00
- **scipy_version**: 1.17.1
- **seed**: 42
- **sklearn_version**: 1.9.1
- **torch.version**: None

## Config

```yaml
seed: 42
task: beat_tracking
dataset:
  name: synthetic
  split: test
  num_samples: 8
  sr: 22050
  duration: 6.0
  tempo: 120.0
adapter:
  name: dsp_beat
metrics:
- beat_fmeasure
- tempo_mae
```

## Metrics

| Metric | Value |
| --- | --- |
| beat_fmeasure.cemgil | 0.930972 |
| beat_fmeasure.f_measure | 1.000000 |
| beat_fmeasure.n_evaluated | 8 |
| beat_fmeasure.p_score | 1.000000 |
| tempo_mae.n_evaluated | 8 |
| tempo_mae.tempo_mae | 0.185320 |

