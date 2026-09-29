"""Audio tagging task.

Input: audio. Output: ``{"scores": {"tag": 0.0..1.0, ...}}`` (soft scores per
tag). Evaluated with mAP (average precision), macro F1 (thresholded), and
macro AUC (ROC) over the tag vocabulary.
"""

from __future__ import annotations

from ..core.registry import TASKS
from ..core.task import Task


@TASKS.register("tagging")
class TaggingTask(Task):
    name = "tagging"

    @classmethod
    def default_metrics(cls):
        return ["tag_map", "tag_macro_f1", "tag_macro_auc"]
