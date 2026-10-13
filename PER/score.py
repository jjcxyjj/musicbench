"""Audio-only ASR compared with target lyrics: PER and Chinese CER / English WER."""
import os
from pathlib import Path
import sys
from shared import device_name, local_directory, local_file
from PER.phonemes import compare_lyrics, aggregate


def evaluate(rows, options, references=None):
    from faster_whisper import WhisperModel
    model_path = local_directory(options.get('model'), 'per.model', weights=False)
    local_file(Path(model_path) / 'model.bin', 'per.model.bin')
    if options.get('espeak_library'):
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = local_file(options['espeak_library'], 'per.espeak_library')
    elif not os.environ.get('PHONEMIZER_ESPEAK_LIBRARY'):
        libraries = sorted((Path(sys.prefix) / 'lib').glob('libespeak-ng.so*'))
        libraries += sorted((Path(sys.prefix) / 'lib').glob('libespeak-ng*.dylib'))
        if libraries:
            os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(libraries[0].resolve())
    device = device_name(options.get('device', 'auto'))
    if device not in ('cpu', 'cuda'):
        raise ValueError('faster-whisper supports cpu/cuda; choose device: cpu on macOS')
    model = WhisperModel(model_path, device=device, local_files_only=True,
                         compute_type=options.get('compute_type', 'float16' if device == 'cuda' else 'int8'))
    results, errors = [], []
    for row in rows:
        try:
            metadata = row['metadata']
            language = metadata.get('language') or options.get('language', 'zh')
            if language not in ('zh', 'en'):
                raise ValueError('PER currently supports zh/en; set language explicitly')
            segments, _ = model.transcribe(row['audio_path'], language=language,
                                          beam_size=int(options.get('beam_size', 5)),
                                          condition_on_previous_text=False, initial_prompt=None, vad_filter=False)
            transcript = ' '.join(segment.text for segment in segments)
            results.append({'id': row['id'], **compare_lyrics(metadata['lyrics'], transcript, language)})
        except Exception as exc:
            errors.append({'id': row['id'], 'reason': str(exc)})
    data = aggregate(results, len(rows), errors)
    status = 'completed' if data['available'] and len(results) == len(rows) else 'partial' if results else 'failed'
    return {'status': status, 'scores': {key: data[key] for key in ('per', 'cer', 'wer')},
            'per_sample': results, 'errors': errors, 'n_expected': len(rows), 'n_evaluated': len(results),
            'reason': data.get('reason'), 'protocol': data['protocol'],
            'model': {'asr': model_path, 'audio_source': 'final output mix; no source separation', 'device': device}}
