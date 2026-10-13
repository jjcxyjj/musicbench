"""Style description + final audio -> MuQ-MuLan similarity."""
import math
from shared import device_name, load_mulan, summarize


def evaluate(rows, options, references=None):
    import librosa
    import torch
    device = device_name(options.get('device', 'auto'))
    model = load_mulan(options).to(device).eval()
    results, errors = [], []
    with torch.inference_mode():
        for row in rows:
            try:
                audio, _ = librosa.load(row['audio_path'], sr=24000, mono=True)
                if not audio.size:
                    raise ValueError('Empty audio')
                audio_embedding = model(wavs=torch.from_numpy(audio).unsqueeze(0).to(device))
                text_embedding = model(texts=[row['metadata']['style_prompt']])
                score = float(model.calc_similarity(audio_embedding, text_embedding).reshape(-1)[0].cpu())
                if not math.isfinite(score):
                    raise ValueError('Non-finite similarity')
                results.append({'id': row['id'], 'similarity': score})
            except Exception as exc:
                errors.append({'id': row['id'], 'reason': str(exc)})
    result = summarize(results, ['similarity'], len(rows), errors)
    result['model'] = {'checkpoint': options['model'], 'text_field': 'style_prompt', 'sample_rate': 24000}
    return result
