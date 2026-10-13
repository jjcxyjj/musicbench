"""Distribution FAD using fixed CLAP or MuQ-MuLan audio embeddings, not per-song matching."""
from shared import device_name, load_mulan, local_directory


def frechet_distance(reference, generated):
    import numpy as np
    if reference.ndim != 2 or generated.ndim != 2 or len(reference) < 2 or len(generated) < 2:
        raise ValueError('FAD 至少需要两条生成音频、两条参考音频的二维特征数组。')
    if reference.shape[1] != generated.shape[1] or not np.isfinite(reference).all() or not np.isfinite(generated).all():
        raise ValueError('Invalid embedding dimensions or values')
    def sqrt_psd(matrix):
        values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
        return (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T
    ref_cov = np.atleast_2d(np.cov(reference, rowvar=False))
    gen_cov = np.atleast_2d(np.cov(generated, rowvar=False))
    ref_root = sqrt_psd(ref_cov)
    delta = reference.mean(axis=0) - generated.mean(axis=0)
    return max(0.0, float(delta @ delta + np.trace(ref_cov + gen_cov - 2 * sqrt_psd(ref_root @ gen_cov @ ref_root))))


def evaluate(rows, options, references):
    import librosa
    import numpy as np
    import torch
    if len(rows) < 2 or len(references) < 2:
        raise ValueError('FAD 至少需要两条生成音频和两条 GT 音频。')
    device = device_name(options.get('device', 'auto'))
    embedding = options.get('embedding', 'clap')
    if embedding == 'clap':
        from transformers import ClapModel, ClapProcessor
        path = local_directory(options.get('model'), 'fad.model')
        model = ClapModel.from_pretrained(path, local_files_only=True).to(device).eval()
        processor = ClapProcessor.from_pretrained(path, local_files_only=True)
        sample_rate = 48000
    elif embedding == 'muq_mulan':
        model = load_mulan(options).to(device).eval()
        sample_rate = 24000
    else:
        raise ValueError('fad.embedding must be clap or muq_mulan')
    def embed(paths):
        features = []
        with torch.inference_mode():
            for path in paths:
                audio, _ = librosa.load(path, sr=sample_rate, mono=True)
                if not audio.size or not np.isfinite(audio).all():
                    raise ValueError(f'Invalid audio: {path}')
                if embedding == 'clap':
                    inputs = processor(audios=audio, sampling_rate=sample_rate, return_tensors='pt').to(device)
                    feature = model.get_audio_features(**inputs).cpu().numpy().reshape(-1)
                else:
                    feature = model(wavs=torch.from_numpy(audio).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
                features.append(feature)
        return np.stack(features)
    score = frechet_distance(embed(references), embed([r['audio_path'] for r in rows]))
    return {'status': 'completed', 'scores': {'fad': score}, 'per_sample': [], 'errors': [],
            'n_expected': len(rows), 'n_evaluated': len(rows),
            'model': {'embedding': embedding + ' audio features', 'checkpoint': options['model'],
                      'n_reference': len(references), 'sample_rate': sample_rate,
                      'note': 'Different embedding models define different FAD protocols; do not compare across encoders.'}}
