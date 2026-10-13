"""ModelAdapter: the interface every model must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .data import Prediction, Sample


class ModelAdapter(ABC):
    """Wraps a model so the runner can call it uniformly.

    Subclasses implement :meth:`load` (build/restore the model) and
    :meth:`predict` (produce a :class:`Prediction` from a :class:`Sample`).
    """

    #: Short name used by the registry (override in subclasses).
    name: str = "base_adapter"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def load(self) -> None:
        """Prepare the model (weights, device). Called once before prediction."""

    @abstractmethod
    def predict(self, sample: Sample) -> Prediction:
        """Run inference on one sample."""

    def model_identity(self) -> Dict[str, Any]:
        """Optional reproducible metadata (weights hash, checkpoint path, ...).

        The runner merges this into ``env.json`` under ``model``.
        """
        return {}


# Common adapter mixins below keep task adapters DRY without adding deps.
