"""Final audio -> SongEval five-dimensional aesthetic scores."""
import importlib.util
from pathlib import Path
from shared import device_name, load_muq, local_file, summarize


def evaluate(rows, options, references=None):
    import librosa
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    repo = Path(options['repo'])
    source = local_file(repo / 'model.py', 'songeval.model.py')
    checkpoint = local_file(repo / 'ckpt/model.safetensors', 'songeval.checkpoint')
    spec = importlib.util.spec_from_file_location('upstream_songeval', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = OmegaConf.to_container(OmegaConf.load(repo / 'config.yaml'))['generator']
    config.pop('_target_', None)
    device = device_name(options.get('device', 'auto'))
    model = module.Generator(**config)
    model.load_state_dict(load_file(checkpoint, device='cpu'), strict=True)
    model = model.to(device).eval()
    encoder = load_muq(options['muq_model']).to(device).eval()
    keys = ['coherence', 'musicality', 'memorability', 'clarity', 'naturalness']
    results, errors = [], []
    with torch.inference_mode():
        for row in rows:
            try:
                audio, _ = librosa.load(row['audio_path'], sr=24000, mono=True)
                if not audio.size or not np.isfinite(audio).all():
                    raise ValueError('Empty or non-finite audio')
                features = encoder(torch.from_numpy(audio).unsqueeze(0).to(device),
                                   output_hidden_states=True)['hidden_states'][6]
                values = model(features).squeeze(0).cpu().tolist()
                if len(values) != 5 or not np.isfinite(values).all():
                    raise ValueError('Invalid SongEval scores')
                scores = dict(zip(keys, values))
                if row['metadata'].get('has_vocal') is not True:
                    scores['naturalness'] = None
                results.append({'id': row['id'], **scores})
            except Exception as exc:
                errors.append({'id': row['id'], 'reason': str(exc)})
    result = summarize(results, keys, len(rows), errors)
    result['model'] = {'repo': str(repo), 'checkpoint': checkpoint, 'muq_model': options['muq_model'], 'device': device}
    return result
