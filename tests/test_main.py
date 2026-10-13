import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import main


def input_file(tmp_path, groups=('ace',)):
    records = []
    for group in groups:
        for index in range(2):
            audio = tmp_path / f'{group}-{index}.wav'
            audio.write_bytes(b'existing audio')
            records.append({'id': str(index), 'algorithm': group, 'audio_path': audio.name,
                            'has_vocal': index == 0, 'lyrics': '目标歌词' if index == 0 else '',
                            'Qwen_caption': 'piano music', 'new_annotation': {'v': index}})
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(records))
    return path


def test_loads_generated_audio_and_preserves_annotations(tmp_path):
    samples, refs = main.load_input(input_file(tmp_path))
    assert all(Path(s['audio_path']).is_absolute() for s in samples)
    assert samples[0]['metadata']['style_prompt'] == 'piano music'
    assert samples[0]['metadata']['new_annotation'] == {'v': 0}
    assert refs == []


def test_rejects_original_dataset_path_as_generated_audio(tmp_path):
    path = tmp_path / 'wrong.json'
    path.write_text('[{"path":"original.wav"}]')
    with pytest.raises(ValueError, match='算法输出音频'):
        main.load_input(path)


def test_duplicate_ids_only_invalid_within_same_algorithm(tmp_path):
    samples, _ = main.load_input(input_file(tmp_path, ('a', 'b')))
    assert len(samples) == 4
    path = tmp_path / 'input.json'
    rows = json.loads(path.read_text())
    rows[1]['id'] = rows[0]['id']
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='必须唯一'):
        main.load_input(path)


def test_interactive_questions_control_optional_metrics(tmp_path, monkeypatch):
    samples, refs = main.load_input(input_file(tmp_path))
    prompts, answers = [], iter(['n', 'n', 'y'])
    def answer(prompt):
        prompts.append(prompt)
        return next(answers)
    monkeypatch.setattr('builtins.input', answer)
    flags, refs = main.choose_inputs(samples, refs)
    assert flags == {'lyrics': False, 'gt_audio': False, 'style_prompt': True}
    assert len(prompts) == 3
    assert main.select_rows('ace_transcriber', samples, flags, refs)[0] == []
    assert len(main.select_rows('muq_mulan', samples, flags, refs)[0]) == 2


def test_lyrics_only_scores_explicit_vocals_with_target_lyrics(tmp_path):
    samples, _ = main.load_input(input_file(tmp_path))
    samples[1]['metadata']['lyrics'] = 'even with lyrics this is instrumental'
    samples += [{'id': 'unknown', 'metadata': {'lyrics': 'unknown voice', 'style_prompt': ''}}]
    rows, _ = main.select_rows('ace_transcriber', samples, {'lyrics': True}, [])
    assert [s['id'] for s in rows] == ['0']
    assert main.select_rows('per', samples, {'lyrics': True}, [])[0] == rows


def test_plan_only_loads_no_models_and_respects_available_inputs(tmp_path, monkeypatch):
    samples, refs = main.load_input(input_file(tmp_path))
    flags, refs = main.choose_inputs(samples, refs, interactive=False)
    monkeypatch.setattr(main, 'load_scorer', lambda _: pytest.fail('Plan must not load scorers'))
    output = tmp_path / 'plan'
    assert main.run(samples, refs, flags, {'enabled_metrics': list(main.SCORERS)}, output, plan_only=True) == 0
    result = json.loads((output / 'scores.json').read_text())
    assert result['status'] == 'planned'
    assert result['algorithms']['ace']['fad']['status'] == 'not_applicable'
    assert result['algorithms']['ace']['ace_transcriber']['eligible_ids'] == ['0']
    with pytest.raises(FileExistsError):
        main.run(samples, refs, flags, {}, output, plan_only=True)


def test_serial_scoring_groups_and_partial_failure_preserve_report(tmp_path, monkeypatch):
    samples, refs = main.load_input(input_file(tmp_path, ('ace', 'other')))
    flags, refs = main.choose_inputs(samples, refs, interactive=False)
    monkeypatch.setattr(main, 'enforce_offline', lambda: None)
    calls = []
    def loader(name):
        def evaluate(rows, options, references):
            calls.append((name, rows[0]['algorithm'], [r['id'] for r in rows]))
            if name == 'songeval' and rows[0]['algorithm'] == 'other':
                raise RuntimeError('missing local weight')
            return {'status': 'completed', 'scores': {'quality': 4.0},
                    'per_sample': [{'id': r['id'], 'quality': 4.0} for r in rows], 'errors': []}
        return evaluate
    monkeypatch.setattr(main, 'load_scorer', loader)
    output = tmp_path / 'results'
    assert main.run(samples, refs, flags, {}, output) == 1
    assert calls == [('songeval', 'ace', ['0', '1']), ('audiobox', 'ace', ['0', '1']),
                     ('songeval', 'other', ['0', '1']), ('audiobox', 'other', ['0', '1'])]
    result = json.loads((output / 'scores.json').read_text())
    assert result['algorithms']['other']['songeval']['status'] == 'failed'
    assert result['algorithms']['other']['audiobox']['scores']['quality'] == 4
    assert (output / 'scores.csv').is_file()
    assert (output / 'songeval.jpg').is_file()
    assert 'missing local weight' in (output / 'report.md').read_text()


def test_fad_uses_distribution_distance_with_no_fake_one_sample_score():
    module = main.load_scorer('fad').__globals__
    distance = module['frechet_distance']
    reference = np.array([[0., 0.], [1., 1.], [2., 0.]])
    assert distance(reference, reference) == pytest.approx(0, abs=1e-7)
    assert distance(reference, reference + 1) == pytest.approx(2)
    with pytest.raises(ValueError, match='至少需要'):
        distance(reference[:1], reference)


def test_audiobox_official_interface_in_current_process(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.touch()
    calls = []
    class Predictor:
        def forward(self, rows):
            calls.append(rows)
            return [{'CE': 6., 'CU': 5., 'PC': 4., 'PQ': 7.}]
    def initialize_predictor(ckpt):
        assert ckpt == str(checkpoint)
        return Predictor()
    monkeypatch.setitem(sys.modules, 'audiobox_aesthetics', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'audiobox_aesthetics.infer', SimpleNamespace(initialize_predictor=initialize_predictor))
    result = main.load_scorer('audiobox')([{'id': 'one', 'audio_path': 'output.wav'}], {'checkpoint': str(checkpoint)})
    assert result['scores']['CE'] == 6
    assert result['status'] == 'completed'
    assert calls == [[{'path': 'output.wav'}]]


def test_gt_paths_are_checked_only_when_selected(tmp_path, monkeypatch):
    samples, _ = main.load_input(input_file(tmp_path))
    answers = iter(['n', 'n', 'n'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    flags, refs = main.choose_inputs(samples, ['/nonexistent/reference.wav'])
    assert refs == []
    assert flags['gt_audio'] is False


def test_ace_captioner_and_transcriber_are_separate_capabilities(monkeypatch):
    import ACE.inference
    calls = []
    class Runner:
        def __init__(self, options):
            self.path = '/local/weights'
        def predict(self, audio, task):
            calls.append((audio, task))
            return 'A piano solo.' if task == 'captioner' else '# Languages\nzh\n# Lyrics\n[Verse]\n春风'
    monkeypatch.setattr(ACE.inference, 'AudioAnnotator', Runner)
    rows = [{'id': 'x', 'audio_path': '/generated/output.wav',
             'metadata': {'lyrics': '春风', 'language': 'zh'}}]
    caption = main.load_scorer('ace_captioner')(rows, {}, [])
    assert caption['per_sample'][0]['caption'] == 'A piano solo.'
    assert caption['scores'] == {}
    transcript = main.load_scorer('ace_transcriber')(rows, {}, [])
    assert transcript['scores']['cer'] == 0
    assert transcript['per_sample'][0]['transcript'] == '[Verse]\n春风'
    assert calls == [('/generated/output.wav', 'captioner'), ('/generated/output.wav', 'transcriber')]


def test_ace_transcriber_rejects_bad_format_and_preserves_raw(monkeypatch):
    import ACE.inference
    class Runner:
        path = '/local/weights'
        def __init__(self, options): pass
        def predict(self, audio, task): return 'unstructured response'
    monkeypatch.setattr(ACE.inference, 'AudioAnnotator', Runner)
    result = main.load_scorer('ace_transcriber')(
        [{'id': 'x', 'audio_path': 'output.wav', 'metadata': {'lyrics': '春风'}}], {}, [])
    assert result['status'] == 'failed'
    assert result['errors'][0]['raw_output'] == 'unstructured response'
    assert result['scores']['cer'] is None


def test_ace_omni_local_loading_audio_only_and_text_generation(tmp_path, monkeypatch):
    from ACE.inference import AudioAnnotator
    model_dir = tmp_path/'model'
    model_dir.mkdir()
    (model_dir/'config.json').write_text('{}')
    (model_dir/'model.safetensors').touch()
    audio = tmp_path/'output.wav'
    audio.touch()
    calls = {}
    class Inputs(dict):
        def to(self, value): return self
    class Model:
        device, dtype = 'cpu', 'float32'
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == str(model_dir)
            assert kwargs['local_files_only'] is True
            return cls()
        def disable_talker(self): calls['disable_talker'] = True
        def eval(self): pass
        def generate(self, **kwargs):
            calls['generate'] = kwargs
            return np.array([[1, 2, 3, 4, 5]])
    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs['local_files_only'] is True
            return cls()
        def apply_chat_template(self, conversation, **kwargs):
            calls['conversation'] = conversation
            return 'prompt'
        def __call__(self, **kwargs):
            assert kwargs['audio'] == ['audio-array']
            return Inputs(input_ids=np.array([[1, 2, 3]]))
        def batch_decode(self, ids, **kwargs):
            assert ids.tolist() == [[4, 5]]
            return ['recognized lyrics']
    from contextlib import nullcontext
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        Qwen2_5OmniForConditionalGeneration=Model, Qwen2_5OmniProcessor=Processor))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, 'qwen_omni_utils', SimpleNamespace(
        process_mm_info=lambda *args, **kwargs: (['audio-array'], None, None)))
    runner = AudioAnnotator({'model': str(model_dir)})
    assert runner.predict(str(audio), 'transcriber') == 'recognized lyrics'
    assert calls['disable_talker']
    assert calls['generate']['return_audio'] is False
    assert calls['generate']['do_sample'] is False
    user = calls['conversation'][1]['content']
    assert user == [{'type': 'text', 'text': '*Task* Transcribe this audio in detail'},
                    {'type': 'audio', 'audio': str(audio)}]


def test_per_asr_never_receives_target_lyrics(tmp_path, monkeypatch):
    model_dir = tmp_path/'whisper'
    model_dir.mkdir()
    (model_dir/'config.json').write_text('{}')
    (model_dir/'model.bin').touch()
    calls = []
    class Whisper:
        def __init__(self, path, **kwargs):
            assert kwargs['local_files_only']
        def transcribe(self, audio, **kwargs):
            calls.append((audio, kwargs))
            return [SimpleNamespace(text='春风')], None
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=Whisper))
    evaluate = main.load_scorer('per')
    monkeypatch.setitem(evaluate.__globals__, 'device_name', lambda _: 'cpu')
    result = evaluate([{'id': 'x', 'audio_path': '/generated/output.wav',
                        'metadata': {'lyrics': '春风吹', 'language': 'zh'}}], {'model': str(model_dir)})
    assert result['scores']['cer'] == pytest.approx(1/3)
    assert calls[0][1]['initial_prompt'] is None
    assert calls[0][1]['condition_on_previous_text'] is False
    assert '春风吹' not in str(calls)
