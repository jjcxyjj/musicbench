"""Text-to-music generation task.

Input: text caption (+ optional style/conditions). Output: generated audio.
Evaluated against reference audio (from the dataset) and the caption, using
L0 "basic" objective features and L1 "professional" deep-model metrics
(FAD / KL / CLAP).
"""

from __future__ import annotations

from ..core.registry import TASKS
from ..core.task import Task


@TASKS.register("text2music")
class Text2MusicTask(Task):
    name = "text2music"

    @classmethod
    def default_metrics(cls, tier: str = "basic"):
        if tier == "professional":
            return ["audio_quality", "fad", "kl_div", "clap_score", "songeval"]
        return ["audio_quality"]
