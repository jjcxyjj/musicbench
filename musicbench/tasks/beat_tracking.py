"""Beat tracking task.

Input: audio. Output: ``{"beats": [t0, t1, ...], "tempo": bpm}``.
Evaluated with ``mir_eval.beat`` (F-measure and friends) plus a tempo error.
"""

from __future__ import annotations

from ..core.registry import TASKS
from ..core.task import Task


@TASKS.register("beat_tracking")
class BeatTrackingTask(Task):
    name = "beat_tracking"

    # Metric registry names, in evaluation order.
    metrics = []  # populated lazily to avoid import cycle; see default_metrics.

    @classmethod
    def default_metrics(cls):
        return ["beat_fmeasure", "tempo_mae"]
