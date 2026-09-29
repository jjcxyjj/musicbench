"""Example: a minimal audio-tagging model adapter.

Same contract as ``beat_model.py`` but for the tagging task: ``predict`` returns
``{"scores": {tag: 0..1}}`` soft scores over the tag vocabulary.

Run it directly:

    python examples/tag_model.py
"""

from __future__ import annotations

import numpy as np

from musicbench.adapters.tagging import _spectral_centroid
from musicbench.core.adapter import ModelAdapter
from musicbench.core.data import Prediction, Sample
from musicbench.core.registry import ADAPTERS


@ADAPTERS.register("my_tag_model")
class MyTagModel(ModelAdapter):
    """A trivial spectral-centroid tagger (replace with your real model)."""

    name = "my_tag_model"

    def load(self) -> None:
        # TODO: load your checkpoint / build your model here.
        pass

    def predict(self, sample: Sample) -> Prediction:
        from scipy.io import wavfile

        sr, data = wavfile.read(sample.audio_path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        else:
            data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        seg = data[: min(len(data), sr)]
        centroid = _spectral_centroid(seg, sr)
        # Soft scores: closer to low center -> higher "low" score, and vice versa.
        d_low, d_high = abs(centroid - 150.0), abs(centroid - 2000.0)
        total = d_low + d_high + 1e-9
        return Prediction(
            id=sample.id,
            output={"scores": {"low": float(d_high / total), "high": float(d_low / total)}},
        )


if __name__ == "__main__":
    from musicbench.datasets.synthetic import SyntheticDataset

    sample = SyntheticDataset(num_samples=1, seed=0).load("test")[0]
    model = MyTagModel()
    model.load()
    pred = model.predict(sample)
    print("sample:", sample.id)
    print("ground truth tags:", sample.ground_truth["tags"])
    print("predicted scores:", pred.output["scores"])
