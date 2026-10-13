"""Built-in evaluation tasks."""

from .beat_tracking import BeatTrackingTask
from .tagging import TaggingTask
from .text2music import Text2MusicTask

__all__ = ["BeatTrackingTask", "TaggingTask", "Text2MusicTask"]
