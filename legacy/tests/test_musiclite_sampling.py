import json
from pathlib import Path

from musicbench.datasets.feature_json import FeatureJsonDataset
from musicbench.workflow import prepare_samples
from musicbench.evaluation.catalog import PRIMARY_RULES, resolve_plan
from scripts.package_dataset import package


def make_batches(tmp_path, counts=(30, 3, 4), nested=False):
    root = tmp_path / 'feature'
    for batch, count in enumerate(counts):
        date = root / f'202609{batch + 9:02d}'
        (date / 'audio').mkdir(parents=True)
        rows = []
        for index in range(count):
            (date / 'audio' / f'{index}.wav').write_bytes(f'audio-{batch}-{index}'.encode())
            record = {'id': str(index), 'path': f'audio/{index}.wav', 'Qwen_caption': f'style {batch}-{index}',
                      'has_vocal': True, 'gf_style_name': '国风流行', 'vocal_keywords': ['Singing'],
                      'latent_path': f'/latents/{batch}/{index}.pth', 'new_annotation': {'v': index},
                      'ground_truth': {'future_label': index}}
            if index % 2:
                record.update(lyrics='你好世界', language='zh')
            rows.append(record)
        location = date / ('audio/5_with_latent.json' if nested and batch == 1 else '5_with_latent.json')
        if location.parent.name == 'audio':
            for row in rows:
                row['path'] = Path(row['path']).name
        location.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    return root


def test_balanced_dates_replay_provenance_and_extra_fields(tmp_path):
    root = make_batches(tmp_path, nested=True)
    a = FeatureJsonDataset(root=root, num_samples=9, seed=42)
    selected = a.load()
    b = FeatureJsonDataset(root=root, num_samples=9, seed=42).load()
    assert [s.id for s in selected] == [s.id for s in b]
    assert len({s.id for s in selected}) == 9
    assert [s['selected_count'] for s in a.selection_summary['sources']] == [3, 3, 3]
    for sample in selected:
        assert Path(sample.audio_path).is_file()
        assert sample.metadata['new_annotation'] == sample.metadata['source_record']['new_annotation']
        assert sample.metadata['latent_path'].startswith('/latents/')
        assert sample.ground_truth['future_label'] == sample.metadata['source_record_index']
        assert sample.metadata['style_prompt'] == sample.metadata['Qwen_caption']
    assert all(len(source['sha256']) == 64 for source in a.selection_summary['sources'])


def test_exhausted_small_dates_and_more_requested_than_available(tmp_path):
    root = make_batches(tmp_path, counts=(1, 2, 20))
    ds = FeatureJsonDataset(root=root, num_samples=10, seed=2)
    assert len(ds.load()) == 10
    assert [s['selected_count'] for s in ds.selection_summary['sources']] == [1, 2, 7]
    ds = FeatureJsonDataset(root=root, num_samples=50)
    assert len(ds.load()) == 23
    assert ds.selection_summary['requested_count'] == 50
    assert ds.selection_summary['selected_count'] == 23


def test_stable_ids_across_different_seeds_and_unique_date_ids(tmp_path):
    root = make_batches(tmp_path, counts=(5, 5))
    a = FeatureJsonDataset(root=root, num_samples=10, seed=1).load()
    b = FeatureJsonDataset(root=root, num_samples=10, seed=9, sampling_strategy='pooled').load()
    assert {s.id for s in a} == {s.id for s in b}
    assert len({s.id for s in a}) == 10  # input id repeats across dates
    assert [s.id for s in a] != [s.id for s in b]


def test_root_json_preferred_over_audio_duplicate(tmp_path):
    root = make_batches(tmp_path, counts=(2,))
    (root / '20260909/audio/5_with_latent.json').write_text('invalid duplicate must not be read')
    ds = FeatureJsonDataset(root=root, num_samples=10)
    assert len(ds.load()) == 2
    assert len(ds.selection_summary['sources']) == 1


def test_packaging_and_repetitions_share_one_draw_and_keep_lyrics(tmp_path):
    root = make_batches(tmp_path, counts=(4, 4, 4))
    config = {'task': 'text2music', 'seed': 42, 'repetitions': 2,
              'dataset': {'name': 'feature_json', 'root': str(root), 'num_samples': 9},
              'inputs': {'style_prompt': True, 'reference_audio': False}}
    output = tmp_path / 'experiment'
    manifest, samples, references = prepare_samples(config, output)
    audit = json.loads((output / 'dataset/sampling.json').read_text())
    assert audit['selected_count'] == 9
    assert len(samples) == 18
    assert len({s.metadata['source_id'] for s in samples}) == 9
    assert len(list((output / 'dataset/audio').glob('*.wav'))) == 9
    for row in samples:
        assert Path(row.audio_path).read_bytes().startswith(b'audio-')
        assert row.metadata['source_record']['new_annotation'] == row.metadata['new_annotation']
    plan = resolve_plan(PRIMARY_RULES, samples, config['inputs'], references)
    lyric_ids = {s.id for s in samples if s.metadata.get('lyrics')}
    assert set(plan['lyrics']['eligible_ids']) == lyric_ids
    assert len(lyric_ids) < len(samples)
    assert plan['fad']['status'] == 'not_applicable'
    assert manifest.is_file()


def test_custom_future_lyric_field_and_no_caption_as_lyrics(tmp_path):
    root = make_batches(tmp_path, counts=(2,))
    path = root / '20260909/5_with_latent.json'
    rows = json.loads(path.read_text())
    rows[0]['new_lyrics'] = '未来歌词'
    path.write_text(json.dumps(rows))
    samples = FeatureJsonDataset(root=root, num_samples=2, lyrics_field='new_lyrics').load()
    by_index = {s.metadata['source_record_index']: s for s in samples}
    assert by_index[0].metadata['lyrics'] == '未来歌词'
    assert by_index[1].metadata['lyrics'] == ''


def test_wizard_defaults_to_musiclite_and_switches_cleanly(monkeypatch):
    from musicbench.workflow import interactive, ROOT, validate
    def answer(prompt):
        if 'conda 环境' in prompt:
            return 'current'
        return ''
    monkeypatch.setattr('builtins.input', answer)
    config = interactive(ROOT / 'configs/tasks.yaml', 'text2music')
    assert config['dataset']['root'] == '/data/MusicLite_data/feature'
    assert config['dataset']['sampling_strategy'] == 'date_balanced'
    assert 'path' not in config['dataset']
    validate(config, ROOT)

    def manifest_answer(prompt):
        if '输入数据格式' in prompt:
            return '2'
        return answer(prompt)
    monkeypatch.setattr('builtins.input', manifest_answer)
    config = interactive(ROOT / 'configs/tasks.yaml', 'text2music')
    assert config['dataset']['name'] == 'manifest'
    assert 'root' not in config['dataset']
    assert 'sampling_strategy' not in config['dataset']
    validate(config, ROOT)
