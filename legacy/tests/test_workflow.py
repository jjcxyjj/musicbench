import copy
import json
from pathlib import Path
import sys

import pytest

from musicbench.workflow import ROOT, environment_command, execute, validate
from musicbench.datasets.manifest import ManifestDataset
from musicbench.metrics.songeval import SongEvalMetric
from musicbench.core.data import Sample, Prediction


def config():
    return {'task': 'beat_tracking', 'tier': 'basic', 'seed': 18,
            'dataset': {'name': 'synthetic', 'num_samples': 3, 'duration': 2},
            'metrics': ['tempo_mae'], 'algorithms': [
                {'name': name, 'env': 'current', 'task': 'beat_tracking',
                 'cwd': str(ROOT / 'algorithms/dsp_beat'), 'infer_script': 'infer.py'}
                for name in ('a', 'b')]}


def test_environment_commands_with_spaces():
    assert environment_command('my env', ['infer.py', 'a b.wav'], '/a b/conda') == [
        '/a b/conda', 'run', '--no-capture-output', '-n', 'my env', 'python', 'infer.py', 'a b.wav']
    assert environment_command('/env path', [1])[3:5] == ['-p', '/env path']
    assert environment_command('current', ['infer.py'])[0] == sys.executable


def test_reject_incompatible_metrics_and_duplicate_algorithms():
    cfg = config()
    cfg['metrics'] = ['clap_score']
    with pytest.raises(ValueError, match='incompatible'):
        validate(cfg, ROOT)
    cfg = config()
    cfg['algorithms'][1]['name'] = 'a'
    with pytest.raises(ValueError, match='distinct'):
        validate(cfg, ROOT)


def test_random_manifest_preserves_labels(tmp_path):
    path = tmp_path / 'data.json'
    rows = [{'id': str(i), 'audio_path': f'{i}.wav', 'ground_truth': {'tempo': i},
             'metadata': {'caption': str(i)}} for i in range(20)]
    path.write_text(json.dumps(rows))
    a = ManifestDataset(path, 5, 42).load()
    b = ManifestDataset(path, 5, 42).load()
    c = ManifestDataset(path, 5, 43).load()
    assert [s.id for s in a] == [s.id for s in b]
    assert [s.id for s in a] != [s.id for s in c]
    assert all(s.ground_truth['tempo'] == int(s.id) for s in a)
    assert all(Path(s.audio_path).is_absolute() for s in a)


def test_two_algorithms_end_to_end(tmp_path):
    cfg = validate(config(), ROOT)
    output = tmp_path / 'run with spaces'
    execute(cfg, output)
    result = json.loads((output / 'results/scores.json').read_text())
    assert result['sample_count'] == 3
    assert result['algorithms']['a'] == result['algorithms']['b']
    assert [r['score'] for r in result['ranking']] == [0.5, 0.5]
    assert (output / 'results/comparison.png').stat().st_size > 1000
    assert 'Aggregate leaders/ties: a, b' in (output / 'results/report.md').read_text()
    assert (output / 'config.yaml').is_file()
    with pytest.raises(FileExistsError):
        execute(cfg, output)


def test_failed_inference_records_status(tmp_path):
    cfg = validate(config(), ROOT)
    script = tmp_path / 'fail.py'
    script.write_text('raise RuntimeError("intentional failure")')
    cfg['algorithms'][0]['infer_script'] = str(script)
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):
        execute(cfg, tmp_path / 'failed')
    assert 'status: failed' in (tmp_path / 'failed/status.yaml').read_text()
    assert 'intentional failure' in (tmp_path / 'failed/logs/infer-a.log').read_text()


def test_songeval_conda_bridge(tmp_path, monkeypatch):
    repo = tmp_path / 'Song Eval'
    (repo / 'ckpt').mkdir(parents=True)
    (repo / 'ckpt/model.safetensors').touch()
    monkeypatch.setenv('MUSICBENCH_SONGEVAL_REPO', str(repo))
    monkeypatch.setenv('MUSICBENCH_SONGEVAL_ENV', 'song env')
    monkeypatch.setenv('CONDA', '/custom/conda')
    def worker(command, **kwargs):
        assert command[:6] == ['/custom/conda', 'run', '--no-capture-output', '-n', 'song env', 'python']
        Path(command[command.index('--output') + 1]).write_text(json.dumps({
            'per_sample': [{'id': 'a', 'musicality': 4}], 'errors': [], 'model': {}}))
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('musicbench.metrics.songeval.subprocess.run', worker)
    result = SongEvalMetric().compute([Prediction(id='a', output={'audio_path': 'a.wav'})], [Sample(id='a', audio_path='a.wav')])
    assert result['musicality'] == 4


def test_plugin_metric_failure_retains_report(tmp_path, monkeypatch):
    module = tmp_path / 'custom_bench_metrics.py'
    module.write_text('''from musicbench.core.registry import METRICS
@METRICS.register("custom_missing")
class CustomMetric:
    def compute(self, predictions, samples):
        return {"available": False, "reason": "test dependency unavailable"}
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    cfg = config()
    cfg.update(plugins=['custom_bench_metrics'], metrics=['custom_missing'],
               allowed_metrics=['custom_missing'], require_metrics=True)
    output = tmp_path / 'partial'
    with pytest.raises(RuntimeError, match='Incomplete experiment'):
        execute(validate(cfg, ROOT), output)
    assert (output / 'results/comparison.png').is_file()
    assert 'test dependency unavailable' in (output / 'results/report.md').read_text()
    assert 'status: failed' in (output / 'status.yaml').read_text()
