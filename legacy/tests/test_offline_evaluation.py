import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from musicbench.core.data import Sample
from musicbench.evaluation.catalog import PRIMARY_RULES, resolve_plan
from musicbench.evaluation.offline import local_file, local_directory, mulan_config, load_mulan


def test_instrumentals_do_not_require_lyrics_even_if_declared():
    samples = [Sample('voice', '', {'has_vocal': True, 'lyrics': '你好'}),
               Sample('instrumental', '', {'has_vocal': False}),
               Sample('wrong_lyrics_on_instrumental', '', {'has_vocal': False, 'lyrics': 'should not score'}),
               Sample('unknown', '', {'lyrics': 'unknown annotation'})]
    plan = resolve_plan(PRIMARY_RULES, samples, {'lyrics': True})['lyrics']
    assert plan['eligible_ids'] == ['voice']
    assert len(plan['excluded']) == 3
    assert plan['excluded'][-1]['reason'] == 'unknown vocal annotation'
    only_instrumental = resolve_plan(PRIMARY_RULES, samples[1:3], {'lyrics': True})
    assert only_instrumental['lyrics']['status'] == 'not_applicable'


def test_vocals_missing_declared_lyrics_fail_but_automatic_mode_uses_subset():
    samples = [Sample('v1', '', {'has_vocal': True, 'lyrics': '你好'}),
               Sample('v2', '', {'has_vocal': True}), Sample('i', '', {'has_vocal': False})]
    with pytest.raises(ValueError, match='Declared lyrics'):
        resolve_plan(PRIMARY_RULES, samples, {'lyrics': True})
    assert resolve_plan(PRIMARY_RULES, samples)['lyrics']['eligible_ids'] == ['v1']


def model_directory(path, config=None):
    path.mkdir()
    (path / 'config.json').write_text(json.dumps(config or {}))
    (path / 'model.safetensors').touch()
    return str(path)


def test_local_assets_never_accept_remote_ids_or_missing_files(tmp_path):
    with pytest.raises(ValueError, match='model IDs are not accepted'):
        local_directory('OpenMuQ/MuQ-large-msd-iter', 'model')
    with pytest.raises(ValueError, match='local file missing'):
        local_file(tmp_path / 'missing.pt', 'checkpoint')
    empty = tmp_path / 'empty'
    empty.mkdir()
    (empty / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='weights missing'):
        local_directory(empty, 'model')


def test_mulan_nested_encoders_are_local_without_modifying_source(tmp_path, monkeypatch):
    audio = model_directory(tmp_path / 'muq')
    text = model_directory(tmp_path / 'xlm-roberta-base')
    model = model_directory(tmp_path / 'mulan', {'audio_model': {'name': 'OpenMuQ/remote'},
                                                'text_model': {'name': 'xlm-roberta-base'}})
    original = (Path(model) / 'config.json').read_bytes()
    calls = []
    def from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return 'loaded'
    monkeypatch.setitem(sys.modules, 'muq', SimpleNamespace(MuQMuLan=SimpleNamespace(from_pretrained=from_pretrained)))
    assert load_mulan({'model': model, 'audio_model': audio, 'text_model': text}) == 'loaded'
    assert calls[0][1]['local_files_only'] is True
    assert calls[0][1]['config']['audio_model']['name'] == audio
    assert calls[0][1]['config']['text_model']['name'] == text
    assert (Path(model) / 'config.json').read_bytes() == original
    with pytest.raises(ValueError, match='local directory missing'):
        mulan_config({'model': model})


def test_worker_offline_guard_blocks_socket_without_connecting():
    code = '''from musicbench.evaluation.offline import enforce_offline
import socket, os
enforce_offline()
assert os.environ['HF_HUB_OFFLINE'] == '1'
s = socket.socket()
try:
    s.connect(('127.0.0.1', 9))
except RuntimeError as e:
    assert 'network access disabled' in str(e)
else:
    raise AssertionError('Networking must be disabled')
finally:
    s.close()
'''
    subprocess.run([sys.executable, '-c', code], check=True, cwd=Path(__file__).resolve().parents[1])


def test_demucs_uses_local_repo_and_legacy_checkpoint_compatibility(tmp_path, monkeypatch):
    from musicbench.evaluation import primary
    repo = tmp_path / 'weights'
    repo.mkdir()
    monkeypatch.setattr(primary, 'device_name', lambda _: 'cpu')
    def run(command, **kwargs):
        assert command[command.index('--repo') + 1] == str(repo)
        assert kwargs['env']['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] == '1'
        output = Path(command[command.index('-o') + 1]) / 'htdemucs' / 'test' / 'vocals.wav'
        output.parent.mkdir(parents=True)
        output.touch()
    monkeypatch.setattr(primary.subprocess, 'run', run)
    result = primary.separate([{'id': 'a', 'audio_path': 'input.wav'}], {'repo': str(repo)}, tmp_path / 'stems')
    assert result['available'] is True
