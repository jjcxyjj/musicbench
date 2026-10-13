import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from musicbench import setup_metrics as setup
from musicbench.workflow import apply_metric_profile, load_metric_profile


def test_component_dedup_and_lyric_separation():
    assert setup.components_for(['fad', 'clap_score', 'lyrics']) == ['clap', 'lyrics', 'demucs']
    assert setup.components_for(['lyrics'], separation=False) == ['lyrics']
    assert setup.parse_reuse(['clap=/data/env', 'songeval=my_song']) == {'clap': '/data/env', 'songeval': 'my_song'}
    with pytest.raises(ValueError):
        setup.parse_reuse(['fad=my_env'])


def test_dry_run_does_not_run_conda_or_write(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Dry run must not execute commands')
    monkeypatch.setattr(setup.subprocess, 'run', forbidden)
    output = tmp_path / 'metrics.yaml'
    assert setup.main(['--dry-run', '--metrics', 'clap_score', 'fad', '--output', str(output)]) == 0
    assert not output.exists()


def mock_runner(monkeypatch, tmp_path, existing=(), failure=None):
    calls = []
    monkeypatch.setattr(setup, 'ROOT', tmp_path)
    monkeypatch.setattr(setup.shutil, 'which', lambda _: '/conda')
    def run(command, **kwargs):
        calls.append(command)
        if command[1:4] == ['env', 'list', '--json']:
            return SimpleNamespace(stdout=json.dumps({'envs': ['/envs/' + e for e in existing]}))
        if '--request' in command:
            request = json.loads(Path(command[command.index('--request') + 1]).read_text())
            result = {'status': 'failed' if request['component'] == failure else 'ready', 'weights_verified': True,
                      'reason': 'simulated missing dependency'}
            if request['component'] == 'lyrics':
                result.update(model_path='/models/whisper', espeak_library='/envs/lyrics/lib/libespeak-ng.so')
            Path(command[command.index('--output') + 1]).write_text(json.dumps(result))
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(setup.subprocess, 'run', run)
    return calls


def test_creates_shared_clap_once_and_publishes(tmp_path, monkeypatch):
    calls = mock_runner(monkeypatch, tmp_path)
    output = tmp_path / 'metrics.yaml'
    assert setup.main(['--metrics', 'clap_score', 'fad', '--output', str(output), '--cache-dir', str(tmp_path / 'cache')]) == 0
    creations = [c for c in calls if c[1:3] == ['env', 'create']]
    assert len(creations) == 1
    data = yaml.safe_load(output.read_text())
    assert data['metric_environments']['clap_score'] == data['metric_environments']['fad']
    assert data['runtime_environment']['HF_HOME'] == str(tmp_path / 'cache/huggingface')
    assert data['setup_verification']['mode'] == 'load'


def test_reused_environment_never_modified_and_paths_saved(tmp_path, monkeypatch):
    calls = mock_runner(monkeypatch, tmp_path, existing=['my_asr'])
    output = tmp_path / 'metrics.yaml'
    assert setup.main(['--metrics', 'lyrics', '--no-separation', '--reuse', 'lyrics=my_asr', '--output', str(output)]) == 0
    assert not any(c[1:3] == ['env', 'create'] for c in calls)
    data = yaml.safe_load(output.read_text())
    assert data['metric_options']['lyrics']['model'] == '/models/whisper'
    assert data['metric_options']['lyrics']['environment']['PHONEMIZER_ESPEAK_LIBRARY'].endswith('.so')
    assert data['metric_options']['lyrics']['audio_source'] == 'mix'


def test_failure_preserves_original_profile_and_continues(tmp_path, monkeypatch):
    calls = mock_runner(monkeypatch, tmp_path, failure='audiobox')
    output = tmp_path / 'metrics.yaml'
    original = 'metric_options:\n  audiobox:\n    checkpoint: /existing/custom.pt\n'
    output.write_text(original)
    assert setup.main(['--metrics', 'audiobox', 'clap_score', '--output', str(output)]) == 1
    assert output.read_text() == original
    assert len([c for c in calls if '--request' in c]) == 2
    status = json.loads(next((tmp_path / 'work/metric-setup').glob('*/status.json')).read_text())
    assert status['components']['audiobox']['status'] == 'failed'
    assert status['components']['clap']['status'] == 'ready'


def test_offline_check_never_creates_or_clones(tmp_path, monkeypatch):
    calls = mock_runner(monkeypatch, tmp_path)
    assert setup.main(['--metrics', 'songeval', '--check-only', '--output', str(tmp_path / 'metrics.yaml')]) == 1
    assert len(calls) == 1
    assert not (tmp_path / 'metrics.yaml').exists()


def test_reuse_missing_environment_is_error(tmp_path, monkeypatch):
    calls = mock_runner(monkeypatch, tmp_path)
    assert setup.main(['--metrics', 'clap_score', '--reuse', 'clap=missing', '--output', str(tmp_path / 'metrics.yaml')]) == 1
    assert len(calls) == 1


def test_profile_path_resolution_and_experiment_precedence(tmp_path):
    path = tmp_path / 'metrics.yaml'
    path.write_text(yaml.safe_dump({'runtime_environment': {'HF_HOME': './cache'},
                                  'metric_options': {'songeval': {'repo': '../SongEval', 'env': 'song'},
                                                     'lyrics': {'model': './weights'}},
                                  'metric_environments': {'clap_score': 'clap'}}))
    profile = load_metric_profile(path)
    result = apply_metric_profile(profile, {'task': 'text2music', 'metric_options': {'songeval': {'env': 'custom'}}})
    assert result['metric_options']['songeval'] == {'env': 'custom', 'repo': str(tmp_path.parent / 'SongEval')}
    assert result['runtime_environment']['HF_HOME'] == str(tmp_path / 'cache')
    assert result['metric_options']['lyrics']['model'] == str(tmp_path / 'weights')
    assert profile['metric_options']['songeval']['env'] == 'song'


def test_profile_environment_reaches_real_scoring_process(tmp_path, monkeypatch):
    from musicbench.workflow import execute, validate, ROOT
    plugin = tmp_path / 'setup_env_probe_plugin.py'
    plugin.write_text('''import os
from musicbench.core.registry import METRICS
@METRICS.register("setup_env_probe")
class Probe:
    def compute(self, predictions, samples):
        assert os.environ["HF_HOME"] == "/configured/cache"
        assert os.environ["PHONEMIZER_ESPEAK_LIBRARY"] == "/configured/espeak.so"
        return {"probe_score": 1.0}
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    cfg = {'task': 'beat_tracking', 'seed': 18,
           'dataset': {'name': 'synthetic', 'num_samples': 1, 'duration': 2},
           'metrics': ['setup_env_probe'], 'allowed_metrics': ['setup_env_probe'],
           'plugins': ['setup_env_probe_plugin'], 'require_metrics': True,
           'runtime_environment': {'HF_HOME': '/configured/cache'},
           'metric_options': {'setup_env_probe': {'environment': {
               'PHONEMIZER_ESPEAK_LIBRARY': '/configured/espeak.so'}}},
           'algorithms': [{'name': 'probe', 'env': 'current', 'task': 'beat_tracking',
                           'cwd': str(ROOT / 'algorithms/dsp_beat'), 'infer_script': 'infer.py'}]}
    output = tmp_path / 'experiment'
    execute(validate(cfg, ROOT), output)
    scores = json.loads((output / 'results/scores.json').read_text())
    assert scores['algorithms']['probe']['setup_env_probe']['probe_score'] == 1.0


def test_workflow_cli_uses_explicit_profile_and_experiment_overrides(tmp_path, monkeypatch):
    import musicbench.workflow as workflow
    profile = tmp_path / 'metrics.yaml'
    profile.write_text('metric_environments:\n  clap_score: shared_clap\nmetric_options:\n  clap_score:\n    model: official/model\n    device: cpu\n')
    experiment = tmp_path / 'experiment.yaml'
    experiment.write_text('task: text2music\nmetrics: [clap_score]\nmetric_options:\n  clap_score:\n    device: cuda\n')
    captured = []
    monkeypatch.setattr(workflow, 'validate', lambda config, base: captured.append(config) or config)
    assert workflow.main(['--metric-config', str(profile), '--config', str(experiment), '--dry-run']) == 0
    assert captured[0]['metric_environments']['clap_score'] == 'shared_clap'
    assert captured[0]['metric_options']['clap_score'] == {'device': 'cuda', 'model': 'official/model'}
