"""Task: binds a task name to its metric set (and later to adapter contract)."""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Type

from .metric import Metric


class Task(ABC):
    """Describes one evaluation task.

    A task declares which metrics evaluate its predictions. The runner uses a
    task to (a) validate adapter output and (b) know which metrics to run.
    """

    name: str = "base_task"
    #: Metric classes to run by default (overridable via config ``metrics``).
    metrics: List[Type[Metric]] = []

    @classmethod
    def default_metrics(cls, tier: str = "basic") -> List[str]:
        """Return metric registry names for this task.

        ``tier`` selects the metric tier: "basic" (no deep models) or
        "professional" (all metrics). Subclasses may override to honor it.
        """
        return [m.name for m in cls.metrics]

    @classmethod
    def describe(cls) -> Dict[str, Any]:
        return {"name": cls.name, "metrics": cls.default_metrics()}
