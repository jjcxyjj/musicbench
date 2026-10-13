"""Audiobox: final audio -> CE/CU/PC/PQ."""
import math
from shared import local_file, summarize

def evaluate(rows, options, references=None):
    from audiobox_aesthetics.infer import initialize_predictor
    predictor = initialize_predictor(ckpt=local_file(options.get('checkpoint'), 'audiobox.checkpoint'))
    if options.get('device', 'auto') != 'auto':
        import torch
        predictor.device = torch.device(options['device'])
        predictor.model = predictor.model.to(predictor.device).eval()
    results, errors = [], []
    keys = ('CE', 'CU', 'PC', 'PQ')
    for row in rows:
        try:
            values = predictor.forward([{'path': row['audio_path']}])[0]
            values = {key: float(values[key]) for key in keys}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError('Non-finite Audiobox score')
            results.append({'id': row['id'], **values})
        except Exception as exc:
            errors.append({'id': row['id'], 'reason': str(exc)})
    out = summarize(results, keys, len(rows), errors)
    out['model'] = {'checkpoint': options.get('checkpoint') or 'official default',

                    'device': str(getattr(predictor, 'device', 'library auto'))}
    return out
