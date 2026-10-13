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


def profile_file(tmp_path):
    path = tmp_path / 'metrics.yaml'
    path.write_text(yaml.safe_dump({'metric_environments': {'clap_score': 'my_clap', 'fad': 'my_clap', 'lyrics': 'asr'},
                                   'metric_options': {'clap_score': {'model': '/local/clap'},
                                                      'lyrics': {'model': '/local/asr', 'audio_source': 'mix'}}}))
    return path


def mock_runner(monkeypatch, tmp_path, existing=(), fail=False):
    calls = []
    monkeypatch.setattr(setup, 'ROOT', tmp_path)
    monkeypatch.setattr(setup.shutil, 'which', lambda _: '/conda')
    def run(command, **kwargs):
        calls.append(command)
        assert 'create' not in command and 'clone' not in command
        if command[1:4] == ['env', 'list', '--json']:
            return SimpleNamespace(stdout=json.dumps({'envs': ['/envs/' + e for e in existing]}))
        assert command[1] == 'run'
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        result = {'status': 'failed' if fail else 'ready', 'weights_verified': not fail}
        if fail:
            result.update(reason='missing libraries', missing_dependencies=['transformers'])
        Path(command[command.index('--output') + 1]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(setup.subprocess, 'run', run)
    return calls


def test_dry_run_does_not_run_conda_or_write(tmp_path, monkeypatch):
    path = profile_file(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Dry run must not execute commands')
    monkeypatch.setattr(setup.subprocess, 'run', forbidden)
    before = path.read_bytes()
    assert setup.main(['--dry-run', '--metrics', 'clap_score', 'fad', '--profile', str(path)]) == 0
    assert path.read_bytes() == before


def test_default_check_never_installs_and_preserves_profile(tmp_path, monkeypatch):
    path = profile_file(tmp_path)
    before = path.read_bytes()
    calls = mock_runner(monkeypatch, tmp_path, existing=['my_clap'])
    assert setup.main(['--metrics', 'clap_score', 'fad', '--profile', str(path)]) == 0
    assert len(calls) == 2  # env inventory, one shared CLAP check
    assert path.read_bytes() == before
    status = json.loads(next((tmp_path / 'work/metric-check').glob('*/status.json')).read_text())
    assert status['offline'] is True
    assert status['components']['clap']['status'] == 'ready'


def test_missing_environment_prints_manual_setup_instead_of_creating(tmp_path, monkeypatch, capsys):
    path = profile_file(tmp_path)
    calls = mock_runner(monkeypatch, tmp_path)
    assert setup.main(['--metrics', 'clap_score', '--profile', str(path)]) == 1
    assert len(calls) == 1
    output = capsys.readouterr().out
    assert 'envs/metric-clap.yml' in output
    assert 'docs/METRIC_SETUP.md' in output
    assert 'https://github.com/huggingface/transformers' in output


def test_missing_libraries_preserve_config_and_show_names(tmp_path, monkeypatch, capsys):
    path = profile_file(tmp_path)
    before = path.read_bytes()
    mock_runner(monkeypatch, tmp_path, existing=['my_clap'], fail=True)
    assert setup.main(['--metrics', 'clap_score', '--profile', str(path)]) == 1
    assert path.read_bytes() == before
    assert 'transformers' in capsys.readouterr().out


def test_missing_profile_never_runs_conda(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('No profile: must not run any command')
    monkeypatch.setattr(setup.subprocess, 'run', forbidden)
    assert setup.main(['--profile', str(tmp_path / 'missing.yaml')]) == 1


def test_init_config_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, 'ROOT', tmp_path)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/metrics.offline.example.yaml').write_text('version: 1\n')
    target = tmp_path / 'local.yaml'
    assert setup.main(['--init-config', '--profile', str(target)]) == 0
    assert target.read_text() == 'version: 1\n'
    assert setup.main(['--init-config', '--profile', str(target)]) == 1
    assert target.read_text() == 'version: 1\n'


def test_profile_path_resolution_and_experiment_precedence(tmp_path):
    path = tmp_path / 'metrics.yaml'
    path.write_text(yaml.safe_dump({'runtime_environment': {'HF_HOME': './cache'},
                                  'metric_options': {'songeval': {'repo': '../SongEval', 'env': 'song', 'muq_model': './muq'},
                                                     'lyrics': {'model': './weights', 'separation': {'repo': './demucs'}}},
                                  'metric_environments': {'clap_score': 'clap'}}))
    profile = load_metric_profile(path)
    result = apply_metric_profile(profile, {'task': 'text2music', 'metric_options': {'songeval': {'env': 'custom'}}})
    assert result['metric_options']['songeval']['env'] == 'custom'
    assert result['metric_options']['songeval']['muq_model'] == str(tmp_path / 'muq')
    assert result['metric_options']['lyrics']['separation']['repo'] == str(tmp_path / 'demucs')
    assert result['runtime_environment']['HF_HOME'] == str(tmp_path / 'cache')
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


def test_shared_environment_selection_clears_old_overrides_without_touching_algorithms():
    from musicbench.workflow import use_shared_environment, evaluation_environment
    original = {'algorithms': [{'env': 'generator'}], 'evaluation_environment': 'old',
                'metric_environments': {'clap_score': 'old_clap'},
                'metric_options': {'songeval': {'env': 'old_song', 'repo': '/song'},
                                   'lyrics': {'separation': {'env': 'old_demucs', 'repo': '/demucs'}}}}
    shared = use_shared_environment(original, 'shared_song')
    assert shared['algorithms'][0]['env'] == 'generator'
    assert evaluation_environment(shared, 'clap_score') == 'shared_song'
    assert evaluation_environment(shared, 'songeval') == 'shared_song'
    assert evaluation_environment(shared, 'separation', shared['metric_options']['lyrics']['separation']) == 'shared_song'
    assert original['metric_options']['songeval']['env'] == 'old_song'
    assert shared['metric_options']['songeval']['repo'] == '/song'


def test_shared_environment_allows_specific_exception():
    from musicbench.workflow import evaluation_environment
    config = {'evaluation_environment': 'shared', 'metric_environments': {'fad': 'fad_special'},
              'metric_options': {'songeval': {'env': 'song_special'}}}
    assert evaluation_environment(config, 'audiobox') == 'shared'
    assert evaluation_environment(config, 'fad') == 'fad_special'
    assert evaluation_environment(config, 'songeval') == 'song_special'


def test_checker_can_force_existing_profile_into_one_environment(tmp_path, monkeypatch):
    path = profile_file(tmp_path)
    before = path.read_bytes()
    calls = mock_runner(monkeypatch, tmp_path, existing=['one_shared'])
    assert setup.main(['--metrics', 'clap_score', 'lyrics', '--profile', str(path), '--shared-env', 'one_shared']) == 0
    workers = [c for c in calls if '--request' in c]
    assert len(workers) == 2
    assert all(c[4] == 'one_shared' for c in workers)
    assert path.read_bytes() == before


def test_wizard_shared_choice_is_not_overridden_by_old_profile(tmp_path, monkeypatch):
    import musicbench.workflow as workflow
    path = profile_file(tmp_path)
    saved = []
    def wizard(catalog, task, profile):
        return {**workflow.use_shared_environment(profile, 'new_shared'),
                'task': 'text2music', 'metrics': ['clap_score']}
    monkeypatch.setattr(workflow, 'interactive', wizard)
    monkeypatch.setattr(workflow, 'validate', lambda config, base: config)
    monkeypatch.setattr(workflow, 'save_yaml', lambda path, config: saved.append(config))
    assert workflow.main(['--metric-config', str(path), '--dry-run']) == 0
    assert saved[0]['metric_environments'] == {}
    assert saved[0]['evaluation_environment'] == 'new_shared'


def test_current_shared_environment_needs_no_conda_inventory(tmp_path, monkeypatch):
    import sys
    path = profile_file(tmp_path)
    monkeypatch.setattr(setup, 'ROOT', tmp_path)
    def run(command, **kwargs):
        assert command[0] == sys.executable
        assert '--request' in command
        Path(command[command.index('--output') + 1]).write_text('{"status":"ready"}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(setup.subprocess, 'run', run)
    assert setup.main(['--metrics', 'clap_score', '--profile', str(path), '--shared-env', 'current',
                       '--conda', '/nonexistent/conda']) == 0
