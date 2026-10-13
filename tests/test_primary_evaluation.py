"""Metric math/protocol and subprocess contracts; no large-model downloads."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from musicbench.core.data import Sample
from musicbench.evaluation.catalog import PRIMARY_RULES, resolve_plan, SCALARS
from musicbench.evaluation.lyrics import edit_alignment, compare_lyrics, aggregate, normalize
from musicbench.workflow import ROOT, validate, execute, prepare_samples, parameter_arguments


def test_metric_inputs_distinguish_lyrics_from_style():
    samples = [Sample(id='x', audio_path='', metadata={'lyrics': '你好', 'has_vocal': True})]
    plan = resolve_plan(PRIMARY_RULES, samples)
    assert plan['audiobox']['status'] == 'scheduled'
    assert plan['lyrics']['status'] == 'scheduled'
    assert plan['clap_score']['status'] == 'not_applicable'
    assert plan['mulan']['status'] == 'not_applicable'
    assert plan['fad']['status'] == 'not_applicable'
    with pytest.raises(ValueError, match='Declared style_prompt'):
        resolve_plan(PRIMARY_RULES, samples, {'style_prompt': True})
    assert SCALARS['audiobox.PC']['direction'] == 'descriptive'


@pytest.mark.parametrize('ref,hyp,counts', [
    ('abc', 'abc', (0, 0, 0)), ('abc', 'adc', (1, 0, 0)),
    ('abc', 'ac', (0, 1, 0)), ('abc', 'abbc', (0, 0, 1)),
    ('abc', '', (0, 3, 0))])
def test_edit_operations(ref, hyp, counts):
    out = edit_alignment(list(ref), list(hyp))
    assert tuple(out[k] for k in ('substitutions', 'deletions', 'insertions')) == counts
    assert out['error_rate'] == sum(counts) / 3


def test_normalization_and_micro_aggregation():
    assert normalize('[Verse 1]\nHello, WORLD!') == 'hello world'
    # Identity G2P is only a fixture; real G2P protocol has separate tests.
    a = {'id': 'a', **compare_lyrics('一二', '一', 'zh', lambda t, l: list(t))}
    b = {'id': 'b', **compare_lyrics('三四五六', '三四五六', 'zh', lambda t, l: list(t))}
    result = aggregate([a, b], 2)
    assert result['cer'] == pytest.approx(1 / 6)
    assert result['per'] == pytest.approx(1 / 6)
    assert result['wer'] is None
    assert a['lexical_alignment']['deletions'] == 1


def test_per_failure_retains_lexical_result():
    def failure(*args):
        raise RuntimeError('g2p unavailable')
    row = {'id': 'a', **compare_lyrics('hello world', 'hello', 'en', failure)}
    result = aggregate([row], 1)
    assert result['wer'] == 0.5
    assert result['per'] is None
    assert result['available'] is False
    assert row['per_error'] == 'g2p unavailable'


def test_real_chinese_g2p():
    pytest.importorskip('pypinyin')
    from musicbench.evaluation.lyrics import phonemes
    assert phonemes('你好', 'zh') == ['n', 'i3', 'h', 'ao3']
    assert phonemes('', 'zh') == []
    with pytest.raises(ValueError, match='mixed scripts'):
        phonemes('你好ABC', 'zh')


def test_audiobox_official_bridge(monkeypatch, tmp_path):
    from musicbench.evaluation.primary import audiobox
    calls = []
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.touch()
    class Predictor:
        def forward(self, rows):
            calls.append(rows)
            if rows[0]['path'] == 'bad':
                raise RuntimeError('decode failed')
            return [{'CE': 7, 'CU': 8, 'PC': 6, 'PQ': 9}]
    def initialize_predictor(ckpt=None):
        assert ckpt == str(checkpoint)
        return Predictor()
    monkeypatch.setitem(sys.modules, 'audiobox_aesthetics', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'audiobox_aesthetics.infer', SimpleNamespace(initialize_predictor=initialize_predictor))
    result = audiobox([{'id': 'a', 'audio_path': 'good'}, {'id': 'b', 'audio_path': 'bad'}], {'checkpoint': str(checkpoint)})
    assert result['CE'] == 7
    assert result['coverage'] == 0.5
    assert calls == [[{'path': 'good'}], [{'path': 'bad'}]]


def test_asr_does_not_receive_target_lyrics(monkeypatch, tmp_path):
    from musicbench.evaluation import primary
    monkeypatch.setattr(primary, 'device_name', lambda _: 'cpu')
    calls = []
    (tmp_path / 'config.json').write_text('{}')
    class Model:
        def __init__(self, *args, **kwargs):
            pass
        def transcribe(self, audio, **kwargs):
            calls.append((audio, kwargs))
            assert kwargs['initial_prompt'] is None
            assert kwargs['condition_on_previous_text'] is False
            assert not any('secret target' in str(value) for value in kwargs.values())
            return iter([SimpleNamespace(text='hello world')]), None
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=Model))
    # A phonemizer fixture avoids requiring the system espeak library in this unit test.
    monkeypatch.setitem(sys.modules, 'phonemizer', SimpleNamespace(phonemize=lambda text, **kw: text))
    monkeypatch.setitem(sys.modules, 'phonemizer.separator', SimpleNamespace(Separator=lambda **kw: None))
    result = primary.lyrics([{'id': 'a', 'audio_path': '/vocals.wav',
                             'metadata': {'lyrics': 'secret target', 'language': 'en'}}], {'device': 'cpu', 'model': str(tmp_path)})
    assert calls[0][0] == '/vocals.wav'
    assert result['wer'] == 1
    assert result['coverage'] == 1


def generation_config(tmp_path):
    inputs = tmp_path / 'inputs.json'
    inputs.write_text(json.dumps([{'id': 'a', 'style_prompt': 'piano solo', 'lyrics': '你好', 'language': 'zh', 'has_vocal': True},
                                 {'id': 'b', 'style_prompt': 'jazz', 'lyrics': '世界', 'language': 'zh', 'has_vocal': False}]))
    return {'version': 2, 'task': 'text2music', 'seed': 5, 'repetitions': 2,
            'dataset': {'name': 'manifest', 'path': str(inputs), 'num_samples': 2},
            'inputs': {'style_prompt': True, 'lyrics': True, 'reference_audio': False},
            'algorithms': [{'name': 'one', 'env': 'current', 'cwd': str(ROOT / 'algorithms/my_t2m'),
                            'infer_script': 'infer.py', 'cuda_visible_devices': 'cpu'}],
            'require_metrics': True}


def test_text_only_manifest_repeats_and_parameter_paths(tmp_path):
    cfg = generation_config(tmp_path)
    cfg['algorithms'][0].update(parameter_schema={'checkpoint': {'type': 'path', 'flag': '--checkpoint'},
                                                'duration': {'type': 'float', 'minimum': 0.1}},
                                parameters={'checkpoint': './weights', 'duration': 1})
    resolved = validate(cfg, tmp_path)
    args = parameter_arguments(resolved['algorithms'][0])
    assert args == ['--checkpoint', str(tmp_path / 'weights'), '--duration', '1.0']
    output = tmp_path / 'prepared'
    output.mkdir()
    manifest, samples, refs = prepare_samples(resolved, output)
    assert len(samples) == 4
    assert all(s.audio_path == '' for s in samples)
    assert {s.metadata['generation_seed'] for s in samples} == {5, 6}
    assert len({s.id for s in samples}) == 4
    assert not refs
    assert all(r['metadata']['style_prompt'] for r in json.loads(manifest.read_text()))


def fake_conda(tmp_path):
    """Fixture external scorers; real generation and all job/report plumbing execute."""
    path = tmp_path / 'fixture-conda'
    path.write_text(f'''#!{sys.executable}
import json,sys,os
from pathlib import Path
args=sys.argv
with open(os.environ['BENCH_TEST_TRACE'],'a') as h: h.write(json.dumps(args)+'\\n')
if 'broken-evaluator' in args:
 print('intentional evaluator failure', file=sys.stderr)
 raise SystemExit(2)
output=Path(args[args.index('--output')+1]); output.parent.mkdir(parents=True,exist_ok=True)
if '--request' in args:
 job=json.loads(Path(args[args.index('--request')+1]).read_text()); metric=job['metric']; rows=job['rows']
 values={{'audiobox':{{'CE':7,'CU':8,'PC':5,'PQ':9}},'clap_score':{{'clap':0.3}},'mulan':{{'similarity':0.4}},'fad':{{'fad':0.1}}}}.get(metric,{{}})
 if metric=='separation':
  result={{'available':True,'coverage':1,'per_sample':[{{'id':r['id'],'audio_path':r['audio_path']}} for r in rows]}}
 elif metric=='lyrics':
  from musicbench.evaluation.lyrics import compare_lyrics,aggregate
  result=aggregate([{{'id':r['id'],**compare_lyrics(r['metadata']['lyrics'],r['metadata']['lyrics'],'zh',lambda t,l:list(t))}} for r in rows],len(rows))
 else:
  result={{'available':True,'coverage':1,**values,'per_sample':[{{'id':r['id'],**values}} for r in rows]}}
else:
 rows=json.loads(Path(args[args.index('--input')+1]).read_text())
 result={{'per_sample':[{{'id':r['id'],'coherence':4,'musicality':4,'memorability':4,'clarity':4,'naturalness':4 if r.get('has_vocal') else None}} for r in rows],'errors':[],'model':{{'fixture':True}}}}
output.write_text(json.dumps(result))
''')
    path.chmod(0o755)
    return path


@pytest.mark.parametrize('break_audiobox,with_reference', [(False, False), (True, False), (False, True)])
def test_generation_full_primary_pipeline(tmp_path, monkeypatch, break_audiobox, with_reference):
    cfg = generation_config(tmp_path)
    if with_reference:
        refs = tmp_path / 'real music'
        refs.mkdir()
        import wave
        for index in range(2):
            with wave.open(str(refs / f'{index}.wav'), 'wb') as handle:
                handle.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
                handle.writeframes(b'\x00\x00' * 800)
        cfg['inputs']['reference_audio'] = True
        cfg['reference_audio'] = {'source': 'directory', 'path': str(refs)}
    cfg['conda'] = str(fake_conda(tmp_path))
    cfg['metric_environments'] = {m: 'fixture-evaluator' for m in PRIMARY_RULES}
    if break_audiobox:
        cfg['metric_environments']['audiobox'] = 'broken-evaluator'
    cfg['metric_options'] = {'songeval': {'env': 'fixture-evaluator'},
                             'lyrics': {'audio_source': 'demucs', 'separation': {'env': 'fixture-evaluator'}}}
    cfg['algorithms'].append({**cfg['algorithms'][0], 'name': 'two'})
    trace = tmp_path / 'trace.jsonl'
    monkeypatch.setenv('BENCH_TEST_TRACE', str(trace))
    cfg = validate(cfg, tmp_path)
    output = tmp_path / 'run'
    if break_audiobox:
        with pytest.raises(RuntimeError, match='Incomplete experiment'):
            execute(cfg, output)
    else:
        execute(cfg, output)
    data = json.loads((output / 'results/scores.json').read_text())
    assert data['ranking'] == []
    assert data['sample_count'] == 4
    if with_reference:
        assert data['algorithms']['one']['fad']['fad'] == 0.1
        assert len(json.loads((output / 'reference_manifest.json').read_text())) == 2
        assert (output / 'results/distribution.png').is_file()
    else:
        assert data['algorithms']['one']['fad']['status'] == 'not_applicable'
    assert data['algorithms']['two']['lyrics']['cer'] == 0
    assert data['algorithms']['two']['mulan']['similarity'] == 0.4
    report = (output / 'results/report.md').read_text()
    assert '歌词匹配度' in report and '风格匹配度' in report
    assert '综合总分' in report
    assert (output / 'results/comparison.png').stat().st_size > 1000
    for name in ('aesthetics.png', 'lyrics.png', 'style.png'):
        assert (output / 'results' / name).is_file()
    commands = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(commands) == (14 if with_reference else 12)
    requests = [json.loads(Path(cmd[cmd.index('--request')+1]).read_text()) for cmd in commands if '--request' in cmd]
    expected = ['audiobox', 'separation', 'lyrics', 'clap_score', 'mulan'] + (['fad'] if with_reference else [])
    assert [r['metric'] for r in requests] == expected * 2
    lyrics_request = next(r for r in requests if r['metric'] == 'lyrics')
    assert len(lyrics_request['rows']) == 2
    assert all(r['metadata']['has_vocal'] is True for r in lyrics_request['rows'])
    assert 'status: ' + ('failed' if break_audiobox else 'completed') in (output / 'status.yaml').read_text()


def test_wizard_asks_inputs_and_parameters_without_tier(tmp_path, monkeypatch):
    from musicbench.workflow import interactive
    prompts = []
    path = tmp_path / 'inputs.json'
    path.write_text('[{"id":"one"}]')
    def answer(prompt):
        prompts.append(prompt)
        if '输入清单 JSON 路径' in prompt:
            return str(path)
        if '清单中有风格描述' in prompt:
            return '2'  # no
        if 'conda 环境' in prompt:
            return 'current'
        if '随机样本数' in prompt:
            return '1'
        return ''
    monkeypatch.setattr('builtins.input', answer)
    result = interactive(ROOT / 'configs/tasks.yaml', 'text2music')
    assert result['inputs'] == {'style_prompt': False, 'lyrics': False, 'reference_audio': False}
    assert 'tier' not in result
    assert not any('验证级别' in p or '选择评价指标' in p for p in prompts)
    assert any('模型权重路径' in p for p in prompts)
    assert any('生成时长' in p for p in prompts)
    assert set(result['metric_options']) == {'audiobox', 'songeval'}
    validate(result, ROOT)


def test_reserved_args_and_nonfinite_parameters_rejected(tmp_path):
    cfg = generation_config(tmp_path)
    cfg['algorithms'][0]['args'] = ['--output=/wrong/path']
    with pytest.raises(ValueError, match='override manifest/output'):
        validate(cfg, tmp_path)
    cfg['algorithms'][0]['args'] = []
    cfg['algorithms'][0].update(parameter_schema={'temperature': {'type': 'float'}}, parameters={'temperature': 'nan'})
    with pytest.raises(ValueError, match='finite'):
        validate(cfg, tmp_path)
