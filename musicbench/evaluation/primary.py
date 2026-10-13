"""Official model bridges. Heavy dependencies are imported only inside workers."""
from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path
import subprocess
import sys


def versions(*packages):
    out = {}
    for package in packages:
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def device_name(requested='auto'):
    import torch
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return requested


def aggregate(rows, dimensions, expected, errors=None):
    out = {'available': bool(rows), 'n_expected': expected, 'n_evaluated': len(rows),
           'coverage': len(rows) / expected if expected else 0, 'per_sample': rows, 'errors': errors or []}
    for name in dimensions:
        vals = [r[name] for r in rows if isinstance(r.get(name), (int, float)) and math.isfinite(r[name])]
        out[name] = sum(vals) / len(vals) if vals else None
    return out


def audiobox(rows, options):
    from audiobox_aesthetics.infer import initialize_predictor
    predictor = initialize_predictor(ckpt=options.get('checkpoint') or None)
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
    out = aggregate(results, keys, len(rows), errors)
    out['model'] = {'checkpoint': options.get('checkpoint') or 'official default',
                    'versions': versions('audiobox_aesthetics', 'torch'),
                    'device': str(getattr(predictor, 'device', 'library auto'))}
    return out


def style(rows, options, name):
    import librosa
    import torch
    import torch.nn.functional as F
    device = device_name(options.get('device', 'auto'))
    results, errors = [], []
    if name == 'mulan':
        from muq import MuQMuLan
        checkpoint = options.get('model', 'OpenMuQ/MuQ-MuLan-large')
        model = MuQMuLan.from_pretrained(checkpoint).to(device).eval()
        sr, key = 24000, 'similarity'
    else:
        from transformers import ClapModel, ClapProcessor
        checkpoint = options.get('model', 'laion/clap-htsat-fused')
        model = ClapModel.from_pretrained(checkpoint).to(device).eval()
        processor = ClapProcessor.from_pretrained(checkpoint)
        sr, key = 48000, 'clap'
    with torch.inference_mode():
        for row in rows:
            try:
                audio, _ = librosa.load(row['audio_path'], sr=sr, mono=True)
                if not len(audio):
                    raise ValueError('Empty audio')
                text = row['metadata']['style_prompt']
                if name == 'mulan':
                    ae = model(wavs=torch.from_numpy(audio).unsqueeze(0).to(device))
                    te = model(texts=[text])
                    value = float(model.calc_similarity(ae, te).reshape(-1)[0].cpu())
                else:
                    ai = processor(audios=audio, sampling_rate=sr, return_tensors='pt').to(device)
                    ti = processor(text=[text], padding=True, return_tensors='pt').to(device)
                    ae, te = model.get_audio_features(**ai), model.get_text_features(**ti)
                    value = float(F.cosine_similarity(ae, te, dim=-1).reshape(-1)[0].cpu())
                if not math.isfinite(value):
                    raise ValueError('Non-finite similarity')
                results.append({'id': row['id'], key: value})
            except Exception as exc:
                errors.append({'id': row['id'], 'reason': str(exc)})
    out = aggregate(results, [key], len(rows), errors)
    out['model'] = {'checkpoint': checkpoint, 'device': device, 'sample_rate': sr,
                    'text_field': 'style_prompt (lyrics excluded)',
                    'versions': versions('muq', 'transformers', 'torch')}
    return out


def fad(rows, options, references):
    import librosa
    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor
    from musicbench.metrics.generation import _fad
    if len(rows) < 2 or len(references) < 2:
        raise ValueError('FAD requires at least two generated and two reference audio files')
    device = device_name(options.get('device', 'auto'))
    checkpoint = options.get('model', 'laion/clap-htsat-fused')
    model = ClapModel.from_pretrained(checkpoint).to(device).eval()
    processor = ClapProcessor.from_pretrained(checkpoint)
    def embed(paths):
        embeddings = []
        with torch.inference_mode():
            for path in paths:
                audio, _ = librosa.load(path, sr=48000, mono=True)
                if not len(audio):
                    raise ValueError(f'Empty audio: {path}')
                inputs = processor(audios=audio, sampling_rate=48000, return_tensors='pt').to(device)
                embeddings.append(model.get_audio_features(**inputs).cpu().numpy().reshape(-1))
        out = np.stack(embeddings)
        if not np.isfinite(out).all():
            raise ValueError('Non-finite audio embeddings')
        return out
    value = _fad(embed(references), embed([r['audio_path'] for r in rows]))
    return {'available': True, 'coverage': 1.0, 'fad': value, 'n_generated': len(rows),
            'n_reference': len(references), 'scope': 'dataset distribution; no pairwise reference required',
            'model': {'embedding': 'CLAP audio features', 'checkpoint': checkpoint, 'device': device,
                      'sample_rate': 48000, 'versions': versions('transformers', 'torch')}}


def separate(rows, options, output_dir):
    """Fixed separation protocol for all algorithms, not native-stem availability."""
    results, errors = [], []
    model = options.get('model', 'htdemucs')
    device = device_name(options.get('device', 'auto'))
    for index, row in enumerate(rows):
        try:
            destination = Path(output_dir) / f'sample_{index:06d}'
            subprocess.run([sys.executable, '-m', 'demucs', '--two-stems', 'vocals', '-n', model,
                            '-d', device, '-o', str(destination), row['audio_path']], check=True,
                           timeout=options.get('timeout', 1800))
            files = list(destination.glob(f'{model}/*/vocals.wav'))
            if len(files) != 1:
                raise ValueError('Demucs did not write one vocal stem')
            results.append({'id': row['id'], 'audio_path': str(files[0].resolve())})
        except Exception as exc:
            errors.append({'id': row['id'], 'reason': str(exc)})
    return {'available': len(results) == len(rows), 'coverage': len(results) / len(rows),
            'per_sample': results, 'errors': errors,
            'model': {'checkpoint': model, 'device': device, 'versions': versions('demucs', 'torch')}}


def lyrics(rows, options):
    from faster_whisper import WhisperModel
    from musicbench.evaluation.lyrics import aggregate as aggregate_lyrics, compare_lyrics
    requested = device_name(options.get('device', 'auto'))
    if requested not in ('cpu', 'cuda'):
        raise ValueError('faster-whisper supports cpu/cuda; use CPU for this worker on macOS')
    checkpoint = options.get('model', 'large-v3')
    model = WhisperModel(checkpoint, device=requested,
                         compute_type=options.get('compute_type', 'float16' if requested == 'cuda' else 'int8'))
    results, errors = [], []
    for row in rows:
        try:
            meta = row['metadata']
            language = meta.get('language') or options.get('language')
            if language not in ('zh', 'en'):
                raise ValueError('Set language=zh/en for lyric evaluation; do not infer it from target lyrics')
            segments, _ = model.transcribe(row['audio_path'], language=language,
                                          beam_size=int(options.get('beam_size', 5)),
                                          condition_on_previous_text=False, initial_prompt=None,
                                          vad_filter=bool(options.get('vad_filter', False)))
            transcript = ' '.join(segment.text for segment in segments)
            results.append({'id': row['id'], **compare_lyrics(meta['lyrics'], transcript, language)})
        except Exception as exc:
            errors.append({'id': row['id'], 'reason': str(exc)})
    out = aggregate_lyrics(results, len(rows), errors)
    out['model'] = {'asr': checkpoint, 'device': requested,
                    'audio_protocol': options.get('audio_source', 'demucs'),
                    'versions': versions('faster-whisper', 'pypinyin', 'phonemizer')}
    return out


HANDLERS = {'audiobox': audiobox, 'lyrics': lyrics}


def compute(job):
    name, rows, options = job['metric'], job['rows'], job.get('options', {})
    if name in ('mulan', 'clap_score'):
        return style(rows, options, name)
    if name == 'fad':
        return fad(rows, options, job.get('reference_paths', []))
    if name == 'separation':
        return separate(rows, options, job['artifact_dir'])
    return HANDLERS[name](rows, options)
