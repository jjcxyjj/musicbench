"""Registry: name -> callable lookup for tasks, metrics, datasets, adapters.

The registry lets YAML configs reference components by short name and lets
users plug in their own implementations without editing framework code.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class Registry:
    """A simple keyed registry with decorator and imperative helpers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, key: str, obj: Any = None) -> Callable[[T], T]:
        """Register ``obj`` under ``key``.

        Usable both as ``@reg.register("k")`` decorator and as
        ``reg.register("k", obj)``.
        """

        def _decorator(target: T) -> T:
            self._items[key] = target
            return target

        if obj is not None:
            self._items[key] = obj
            return obj  # type: ignore[return-value]
        return _decorator

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(f"{self.name}: unknown key {key!r}. Available: {sorted(self._items)}")
        return self._items[key]

    def has(self, key: str) -> bool:
        return key in self._items

    def keys(self):
        return list(self._items.keys())


# Four registries mirroring the four extension points of the harness.
TASKS = Registry("tasks")
METRICS = Registry("metrics")
DATASETS = Registry("datasets")
ADAPTERS = Registry("adapters")


def get_task(name: str) -> Type:
    return TASKS.get(name)


def get_metric(name: str) -> Type:
    return METRICS.get(name)


def get_dataset(name: str) -> Type:
    return DATASETS.get(name)


def get_adapter(name: str) -> Type:
    return ADAPTERS.get(name)
