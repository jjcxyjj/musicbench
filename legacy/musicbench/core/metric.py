"""Metric: a function from (predictions, samples) -> dict of numbers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .data import Prediction, Sample


class Metric(ABC):
    """Computes one or more scalar/metric values over a whole evaluation set."""

    name: str = "base_metric"

    @abstractmethod
    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        """Return a flat dict of metric name -> value.

        Values must be JSON-serializable (float/int/str/bool, no numpy types).
        """
