"""Reproducible cross-date sampling of MusicLite processed feature records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from ..core.data import Sample
from ..core.dataset import Dataset
from ..core.registry import DATASETS


@DATASETS.register('feature_json')
class FeatureJsonDataset(Dataset):
    name = 'feature_json'

    def __init__(self, root='/data/MusicLite_data/feature', num_samples=10, seed=0,
                 json_name='5_with_latent.json', audio_field='path',
                 caption_field='Qwen_caption', lyrics_field='lyrics', language_field='language',
                 sampling_strategy='date_balanced'):
        self.root = Path(root).expanduser().resolve()
        self.num_samples, self.seed = int(num_samples), int(seed)
        if self.num_samples <= 0:
            raise ValueError('num_samples must be positive')
        if sampling_strategy not in ('date_balanced', 'pooled'):
            raise ValueError('sampling_strategy must be date_balanced or pooled')
        self.json_name, self.sampling_strategy = json_name, sampling_strategy
        self.audio_field, self.caption_field = audio_field, caption_field
        self.lyrics_field, self.language_field = lyrics_field, language_field

    def _files(self):
        # Prefer the date-level file; the audio-level file is an alternative layout,
        # not a second copy of the same batch to sample twice.
        files = []
        for date in sorted(self.root.glob('*')):
            if not date.is_dir():
                continue
            direct = date / self.json_name
            nested = date / 'audio' / self.json_name
            if direct.is_file():
                files.append(direct)
            elif nested.is_file():
                files.append(nested)
        if not files:
            raise FileNotFoundError(f'No {self.json_name} under date directories in {self.root}')
        return files

    def load(self, split='test'):
        rng = random.Random(self.seed)
        groups, pool, total, sources = {}, [], 0, []
        # Reservoir sampling retains at most N candidates per date (or N globally),
        # rather than keeping the processed features from every date in memory.
        for path in self._files():
            try:
                raw = path.read_bytes()
                data = json.loads(raw)
            except (OSError, ValueError) as exc:
                raise ValueError(f'Cannot read feature JSON {path}: {exc}') from exc
            digest = hashlib.sha256(raw).hexdigest()
            del raw
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                raise ValueError(f'Expected JSON array or record object: {path}')
            date = path.relative_to(self.root).parts[0]
            candidates = []
            for index, record in enumerate(data):
                if not isinstance(record, dict):
                    raise ValueError(f'Expected record object: {path}, index={index}')
                total += 1
                item = (path, index, record)
                target = candidates if self.sampling_strategy == 'date_balanced' else pool
                seen = index + 1 if self.sampling_strategy == 'date_balanced' else total
                if len(target) < self.num_samples:
                    target.append(item)
                else:
                    position = rng.randrange(seen)
                    if position < self.num_samples:
                        target[position] = item
            groups[date] = candidates
            sources.append({'date': date, 'json_path': str(path), 'sha256': digest,
                            'record_count': len(data), 'selected_count': 0})
        if not total:
            raise ValueError('Feature dataset contains no samples')
        if self.sampling_strategy == 'date_balanced':
            for candidates in groups.values():
                rng.shuffle(candidates)
            selected = []
            while len(selected) < min(self.num_samples, total):
                dates = sorted(date for date, values in groups.items() if values)
                rng.shuffle(dates)
                for date in dates:
                    selected.append(groups[date].pop())
                    if len(selected) == min(self.num_samples, total):
                        break
        else:
            selected = pool
            rng.shuffle(selected)
        samples = [self._to_sample(*item) for item in selected]
        source_map = {s['json_path']: s for s in sources}
        for path, _, _ in selected:
            source_map[str(path)]['selected_count'] += 1
        self.selection_summary = {
            'root': str(self.root), 'strategy': self.sampling_strategy, 'seed': self.seed,
            'requested_count': self.num_samples, 'selected_count': len(samples), 'total_records': total,
            'sources': sources, 'selected_ids': [s.id for s in samples],
            'note': 'Date-balanced rounds randomize date order and choose records without replacement; '
                    'pooled sampling gives every record equal probability. Source JSON must remain fixed for seed replay.'}
        if len(samples) < self.num_samples:
            print(f'[dataset] Requested {self.num_samples}, only {len(samples)} available; using all records.')
        return samples

    def _resolve_audio(self, value, source):
        if not value:
            return ''
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path.resolve())
        date = self.root / source.relative_to(self.root).parts[0]
        # Prefer JSON-relative paths, then a filename in the date's audio folder.
        candidates = [source.parent / path, date / path, date / 'audio' / path]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        return str(candidates[0].resolve())  # workflow reports missing audio explicitly

    def _to_sample(self, source, index, record):
        audio = self._resolve_audio(record.get(self.audio_field) or record.get('path'), source)
        caption = record.get(self.caption_field) or record.get('Qwen_caption') or ''
        date = source.relative_to(self.root).parts[0]
        source_key = hashlib.sha256(str(source.relative_to(self.root)).encode()).hexdigest()[:10]
        # Stable across seeds and order; upstream IDs can be reused in different dates.
        sample_id = f'ml_{date}_{source_key}_{index:08d}'
        metadata = {**record.get('metadata', {}),
                    'caption': caption, 'style_prompt': record.get('style_prompt') or caption,
                    'lyrics': record.get(self.lyrics_field) or '',
                    'language': record.get(self.language_field),
                    'has_vocal': record.get('has_vocal'), 'vocal_score': record.get('vocal_score'),
                    'style_name': record.get('gf_style_name'), 'style_code': record.get('gf_style_code'),
                    # Keep all new feature/annotation fields available to future adapters.
                    **{key: value for key, value in record.items() if key not in ('metadata', 'ground_truth')},
                    'source_record': record, 'source_json': str(source), 'source_date': date,
                    'source_record_index': index, 'source_audio_path': audio}
        # Explicit mapping wins over raw keys with the same name.
        metadata.update(caption=caption, style_prompt=record.get('style_prompt') or caption,
                        lyrics=record.get(self.lyrics_field) or '', language=record.get(self.language_field))
        ground_truth = {**record.get('ground_truth', {}), 'caption': caption, 'has_vocal': record.get('has_vocal')}
        return Sample(id=sample_id, audio_path=audio,
                      metadata={k: v for k, v in metadata.items() if v is not None}, ground_truth=ground_truth)
