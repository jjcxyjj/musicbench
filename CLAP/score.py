"""Style description + final audio -> local Transformers CLAP similarity."""
import math
from shared import device_name, local_directory, summarize


def evaluate(rows, options, references=None):
    import librosa
    import torch
    from torch.nn.functional import cosine_similarity
    from transformers import ClapModel, ClapProcessor
    path = local_directory(options.get('model'), 'clap.model')
    device = device_name(options.get('device', 'auto'))
    model = ClapModel.from_pretrained(path, local_files_only=True).to(device).eval()
    processor = ClapProcessor.from_pretrained(path, local_files_only=True)
    results, errors = [], []
    with torch.inference_mode():
        for row in rows:
            try:
                audio, _ = librosa.load(row['audio_path'], sr=48000, mono=True)
                if not audio.size:
                    raise ValueError('Empty audio')
                audio_input = processor(audios=audio, sampling_rate=48000, return_tensors='pt').to(device)
                text_input = processor(text=[row['metadata']['style_prompt']], padding=True, return_tensors='pt').to(device)
                value = float(cosine_similarity(model.get_audio_features(**audio_input),
                                                model.get_text_features(**text_input)).reshape(-1)[0].cpu())
                if not math.isfinite(value):
                    raise ValueError('Non-finite CLAP score')
                results.append({'id': row['id'], 'clap': value})
            except Exception as exc:
                errors.append({'id': row['id'], 'reason': str(exc)})
    result = summarize(results, ['clap'], len(rows), errors)
    result['model'] = {'checkpoint': path, 'sample_rate': 48000, 'text_field': 'style_prompt'}
    return result
