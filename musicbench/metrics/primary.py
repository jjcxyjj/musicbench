"""Registry bridges for task-defined primary metrics (lazy optional dependencies)."""
from ..core.metric import Metric
from ..core.registry import METRICS


class PrimaryMetric(Metric):
    def compute(self, predictions, samples):
        from ..evaluation.primary import compute
        by_id = {s.id: s for s in samples}
        rows = [{'id': p.id, 'audio_path': p.output.get('audio_path', ''),
                 'metadata': by_id[p.id].metadata} for p in predictions if p.id in by_id]
        try:
            return compute({'metric': self.name, 'rows': rows, 'options': {}})
        except Exception as exc:
            return {'available': False, 'coverage': 0.0, 'reason': str(exc)}


@METRICS.register('audiobox')
class AudioboxMetric(PrimaryMetric):
    name = 'audiobox'


@METRICS.register('mulan')
class MuLanMetric(PrimaryMetric):
    name = 'mulan'


@METRICS.register('lyrics')
class LyricsMetric(PrimaryMetric):
    name = 'lyrics'
