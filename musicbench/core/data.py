"""Data model: Sample and Prediction.

``Sample`` carries everything a model receives (audio path plus optional
metadata). ``Prediction`` carries the model's structured output keyed to a
sample id, so predictions can be serialized to ``predictions.jsonl`` and
evaluated offline (independently of the model that produced them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Sample:
    """One evaluation instance.

    Attributes:
        id: Unique sample identifier (used to join predictions back to labels).
        audio_path: Path to the input audio file (wav/flac/mp3).
        metadata: Free-form side information (e.g. sample rate, duration).
        ground_truth: Task-specific reference labels (beats, tags, chords, ...).
    """

    id: str
    audio_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    ground_truth: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """One structured model output.

    Attributes:
        id: Matches ``Sample.id``.
        output: Task-specific prediction payload (beats/tempo, tag scores, ...).
    """

    id: str
    output: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict (numpy arrays -> lists)."""
        return {"id": self.id, "output": _jsonable(self.output)}


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python types."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def prediction_from_json(data: Dict[str, Any]) -> Prediction:
    return Prediction(id=data["id"], output=dict(data.get("output", {})))
