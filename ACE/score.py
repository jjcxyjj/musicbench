"""Captioner annotations, or transcriber outputs scored against target lyrics."""
import re
from ACE.inference import AudioAnnotator
from PER.phonemes import compare_lyrics, aggregate
from shared import summarize

def extract_lyrics(text):
    match = re.search(r'^#\s*Lyrics\s*$', text, flags=re.I | re.M)
    if not match:
        raise ValueError('Transcriber output lacks # Lyrics; raw output retained in error')
    lyrics = text[match.end():].strip()
    if not lyrics:
        raise ValueError('Transcriber returned empty lyrics')
    return lyrics

def evaluate(rows, options, references=None):
    task = options['task']
    runner = AudioAnnotator(options)
    results, errors = [], []
    for row in rows:
        raw = ''
        try:
            raw = runner.predict(row['audio_path'], task)
            if not raw:
                raise ValueError('Empty model output')
            entry = {'id': row['id'], 'raw_output': raw}
            if task == 'captioner':
                entry['caption'] = raw
            else:
                transcript = extract_lyrics(raw)
                language = row['metadata'].get('language') or options.get('language', 'zh')
                entry.update(compare_lyrics(row['metadata']['lyrics'], transcript, language))
            results.append(entry)
        except Exception as exc:
            errors.append({'id': row['id'], 'reason': str(exc), 'raw_output': raw})
    result = summarize(results, [], len(rows), errors)
    if task == 'transcriber':
        data = aggregate(results, len(rows), errors)
        result['scores'] = {key: data[key] for key in ('per', 'cer', 'wer')}
        result['protocol'] = data['protocol']
        if not data['available'] and results:
            result.update(status='partial', reason=data.get('reason'))
    else:
        result['reason'] = '音乐描述已保存；captioner 本身不产生风格匹配数值'
    result['model'] = {'path': runner.path, 'task': task,
                       'input': 'final audio only; no target lyrics/style supplied'}
    return result
