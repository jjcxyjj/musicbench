import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import musicbench.algorithms as algorithms
from musicbench.workflow import execute, validate


def test_scaffold_is_discovered_and_fails_until_connected(tmp_path):
    directory = tmp_path / 'algorithms'
    destination = algorithms.register('new_model', 'text2music', 'model_env', directory=directory)
    cfg = algorithms.discover('text2music', directory)['new_model']
    assert cfg['env'] == 'model_env'
    assert cfg['cwd'] == str(destination)
    assert cfg['infer_script'] == str(destination / 'infer.py')
    assert set(cfg['parameter_schema']) == {'checkpoint', 'duration', 'steps', 'guidance', 'sr'}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('[{"id":"one","metadata":{"style_prompt":"piano"}}]')
    proc = subprocess.run([sys.executable, cfg['infer_script'], '--manifest', str(manifest),
                           '--output', str(tmp_path / 'predictions.jsonl')], capture_output=True, text=True)
    assert proc.returncode != 0 and 'Connect your real model' in proc.stderr
    assert not (tmp_path / 'predictions.jsonl').exists()
    saved = (destination / 'config.yaml').read_bytes()
    with pytest.raises(FileExistsError):
        algorithms.register('new_model', 'text2music', 'other_env', directory=directory)
    assert (destination / 'config.yaml').read_bytes() == saved


def test_external_script_and_workdir_are_preserved(tmp_path):
    model = tmp_path / 'external model'
    model.mkdir()
    script = model / 'adapter.py'
    script.write_text('print("adapter")')
    schema = {'temperature': {'type': 'float', 'flag': '--temperature', 'default': 0.7}}
    algorithms.register('external', 'text2music', '/env path', script='adapter.py', cwd=str(model),
                        schema=schema, gpu='1', directory=tmp_path / 'registrations')
    cfg = algorithms.discover('text2music', tmp_path / 'registrations')['external']
    assert cfg['cwd'] == str(model)
    assert cfg['infer_script'] == str(script)
    assert cfg['parameter_schema'] == schema
    assert cfg['cuda_visible_devices'] == '1'
    assert script.read_text() == 'print("adapter")'


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'bad name', ''])
def test_bad_names_do_not_create_files(tmp_path, name):
    with pytest.raises(ValueError):
        algorithms.register(name, 'text2music', 'env', directory=tmp_path)
    assert not list(tmp_path.iterdir())


def test_invalid_script_schema_and_task_do_not_register(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        algorithms.register('missing', 'text2music', 'env', script=str(tmp_path / 'missing.py'), directory=tmp_path)
    with pytest.raises(ValueError, match='reserved'):
        algorithms.register('reserved', 'text2music', 'env', schema={'output': {'flag': '--output'}}, directory=tmp_path)
    with pytest.raises(ValueError, match='provide --script'):
        algorithms.register('beat', 'beat_tracking', 'env', directory=tmp_path)
    assert not list(tmp_path.iterdir())


def test_cli_register_and_list(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(algorithms, 'ROOT', tmp_path)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/tasks.yaml').write_text('tasks: {text2music: {}}')
    assert algorithms.main(['add', '--name', 'cli_model', '--task', 'text2music', '--env', 'env']) == 0
    assert (tmp_path / 'algorithms/cli_model/infer.py').is_file()
    assert algorithms.main(['list', '--task', 'text2music']) == 0
    assert 'cli_model: task=text2music env=env' in capsys.readouterr().out
    assert algorithms.main(['add', '--name', 'bad_task', '--task', 'unknown', '--env', 'env']) == 1


def test_external_adapter_runs_with_project_imports(tmp_path, monkeypatch):
    project = tmp_path / 'external project'
    project.mkdir()
    (project / 'local_model.py').write_text('VALUE=42\n')
    adapters = tmp_path / 'adapters'
    adapters.mkdir()
    script = adapters / 'adapter.py'
    script.write_text('''import argparse,json,wave
from pathlib import Path
from local_model import VALUE
p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--output');args=p.parse_args()
assert VALUE==42
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
rows=json.loads(Path(args.manifest).read_text())
with output.open('w') as result:
 for i,row in enumerate(rows):
  path=output.parent/f'{i}.wav'
  with wave.open(str(path),'wb') as h:
   h.setparams((1,2,8000,0,'NONE','not compressed'));h.writeframes(bytes(1600))
  result.write(json.dumps({'id':row['id'],'output':{'audio_path':str(path)}})+'\\n')
''')
    plugins = tmp_path / 'plugins'
    plugins.mkdir()
    (plugins / 'registration_contract.py').write_text('''from musicbench.core.registry import METRICS
@METRICS.register('contract_score')
class Metric:
 def compute(self,predictions,samples):
  return {'available':True,'coverage':1,'score':0.9}
''')
    monkeypatch.syspath_prepend(str(plugins))
    # Nonempty inherited PYTHONPATH must not prevent imports from the external cwd.
    monkeypatch.setenv('PYTHONPATH', str(plugins))
    registry = tmp_path / 'registered'
    algorithms.register('external', 'text2music', 'current', script=str(script), cwd=str(project), directory=registry)
    item = algorithms.discover('text2music', registry)['external']
    inputs = tmp_path / 'inputs.json'
    inputs.write_text('[{"id":"one","style_prompt":"piano"}]')
    cfg = {'task':'text2music','dataset':{'name':'manifest','path':str(inputs),'num_samples':1},
           'algorithms':[item], 'plugins':['registration_contract'],
           'metric_rules':{'contract_score':{'category':'aesthetics','requires':[]}},
           'report_scalars':{'contract_score.score':{'category':'aesthetics','direction':'higher'}}}
    output = tmp_path / 'run'
    execute(validate(cfg, tmp_path), output)
    scores=json.loads((output / 'results/scores.json').read_text())
    assert scores['algorithms']['external']['contract_score']['score']==0.9
    assert (output / 'results/report.md').is_file()
