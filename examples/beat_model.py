"""Example: a minimal beat-tracking model adapter.

This shows the contract every musicbench adapter must implement:
``load()`` + ``predict(sample)``, and how to register it so a YAML config can
reference it by name (``adapter.name: my_beat_model``).

Run it directly to see it produce a prediction on a synthetic clip:

    python examples/beat_model.py
"""

from __future__ import annotations

import numpy as np

from musicbench.adapters.beat import _peak_pick, _read_wav, _rms_envelope, _tempo_from_beats
from musicbench.core.adapter import ModelAdapter
from musicbench.core.data import Prediction, Sample
from musicbench.core.registry import ADAPTERS


@ADAPTERS.register("my_beat_model")
class MyBeatModel(ModelAdapter):
    """A trivial energy-onset beat tracker (replace with your real model)."""

    name = "my_beat_model"

    def load(self) -> None:
        # TODO: load your checkpoint / build your model here.
        pass

    def predict(self, sample: Sample) -> Prediction:
        audio, sr = _read_wav(sample.audio_path)
        env = _rms_envelope(audio, sr)
        beats = _peak_pick(env, sr, min_interval=0.3, threshold_std=0.5)
        tempo = _tempo_from_beats(beats)
        return Prediction(id=sample.id, output={"beats": beats, "tempo": tempo})


if __name__ == "__main__":
    from musicbench.datasets.synthetic import SyntheticDataset

    sample = SyntheticDataset(num_samples=1, seed=0).load("test")[0]
    model = MyBeatModel()
    model.load()
    pred = model.predict(sample)
    print("sample:", sample.id)
    print("ground truth beats:", np.round(sample.ground_truth["beats"], 3).tolist())
    print("predicted beats:", np.round(pred.output["beats"], 3).tolist())
    print("predicted tempo:", pred.output["tempo"])
