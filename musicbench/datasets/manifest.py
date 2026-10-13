"""Randomly sample a user-supplied manifest, preserving labels and metadata."""
import json
from pathlib import Path
import random
from musicbench.core.data import Sample
from musicbench.core.dataset import Dataset
from musicbench.core.registry import DATASETS


@DATASETS.register('manifest')
class ManifestDataset(Dataset):
    def __init__(self, path, num_samples=10, seed=42):
        self.path = Path(path).resolve()
        self.count, self.seed = int(num_samples), int(seed)
        if self.count <= 0:
            raise ValueError('num_samples must be positive')

    def load(self, split='test'):
        rows = json.loads(self.path.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError('Manifest must be a nonempty JSON array')
        ids = [str(row['id']) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate manifest IDs')
        if self.count > len(rows):
            raise ValueError('Requested sample count exceeds manifest size')
        selected = random.Random(self.seed).sample(rows, self.count)
        samples = []
        for row in selected:
            audio = row.get('audio_path') or ''
            if audio:
                path = Path(audio).expanduser()
                if not path.is_absolute():
                    path = self.path.parent / path
                audio = str(path.resolve())
            metadata = dict(row.get('metadata', {}))
            for key in ('style_prompt', 'lyrics', 'language', 'has_vocal', 'caption'):
                if key in row:
                    metadata.setdefault(key, row[key])
            samples.append(Sample(id=str(row['id']), audio_path=audio,
                                  ground_truth=row.get('ground_truth', {}), metadata=metadata))
        return samples
