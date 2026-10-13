"""Dataset: a source of evaluation samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .data import Sample


class Dataset(ABC):
    """Yields :class:`Sample` objects (with ground truth) for a split."""

    name: str = "base_dataset"

    @abstractmethod
    def load(self, split: str = "test") -> List[Sample]:
        """Return the samples for ``split``."""
