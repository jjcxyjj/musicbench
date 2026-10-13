"""Audio-only transcription is scored against normalized text and phonemes."""
from __future__ import annotations

import re
import unicodedata


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    # Structural/performance annotations are not words to be sung.
    text = re.sub(r'\[[^\]]*\]|<[^>]*>', ' ', text)
    text = ''.join(c if c.isalnum() or c.isspace() or c == "'" else ' ' for c in text)
    return ' '.join(text.split())


def edit_alignment(reference, hypothesis):
    """Levenshtein alignment with deterministic substitution/deletion/insertion ties."""
    n, m = len(reference), len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        costs[i][0] = i
    for j in range(m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                              costs[i - 1][j] + 1, costs[i][j - 1] + 1)
    counts = {'hits': 0, 'substitutions': 0, 'deletions': 0, 'insertions': 0}
    differences = []
    i, j = n, m
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            kind = 'hits' if reference[i - 1] == hypothesis[j - 1] else 'substitutions'
            counts[kind] += 1
            if kind != 'hits':
                differences.append({'operation': kind, 'reference': reference[i - 1],
                                    'hypothesis': hypothesis[j - 1], 'reference_index': i - 1})
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            counts['deletions'] += 1
            differences.append({'operation': 'deletions', 'reference': reference[i - 1],
                                'hypothesis': '', 'reference_index': i - 1})
            i -= 1
        else:
            counts['insertions'] += 1
            differences.append({'operation': 'insertions', 'reference': '',
                                'hypothesis': hypothesis[j - 1], 'reference_index': i})
            j -= 1
    return {**counts, 'n_reference': n,
            'error_rate': (counts['substitutions'] + counts['deletions'] + counts['insertions']) / n if n else None,
            'differences': list(reversed(differences))}


def phonemes(text, language):
    if not text:
        return []
    if language == 'zh':
        from pypinyin import pinyin, Style
        initials = pinyin(text, style=Style.INITIALS, strict=True, errors='default')
        finals = pinyin(text, style=Style.FINALS_TONE3, strict=True, neutral_tone_with_five=True, errors='default')
        result = []
        for initial, final in zip(initials, finals):
            # Non-Han tokens require a separate language protocol; do not silently call them phones.
            if not re.fullmatch(r'[a-züv]+[1-5]', final[0]):
                if final[0].strip():
                    raise ValueError('Chinese PER requires Han lyrics; mixed scripts/digits need an explicit G2P protocol')
                continue
            if initial[0]:
                result.append(initial[0])
            result.append(final[0])
        return result
    if language == 'en':
        from phonemizer import phonemize
        from phonemizer.separator import Separator
        result = phonemize(text, language='en-us', backend='espeak', strip=True,
                          separator=Separator(phone=' ', word=' | ', syllable=''),
                          preserve_punctuation=False, with_stress=False)
        return [phone for phone in result.split() if phone != '|']
    raise ValueError(f'PER protocol currently supports zh/en, received {language!r}')


def compare_lyrics(reference, hypothesis, language, g2p=phonemes):
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError('Empty target lyrics after removing annotations')
    if language not in ('zh', 'en'):
        raise ValueError(f'Unsupported lyric language: {language}')
    lexical_name = 'cer' if language == 'zh' else 'wer'
    ref_units = list(ref.replace(' ', '')) if language == 'zh' else ref.split()
    hyp_units = list(hyp.replace(' ', '')) if language == 'zh' else hyp.split()
    lexical = edit_alignment(ref_units, hyp_units)
    result = {'reference': reference, 'transcript': hypothesis, 'normalized_reference': ref,
              'normalized_transcript': hyp, 'language': language, lexical_name: lexical['error_rate'],
              'lexical_alignment': lexical}
    try:
        rp, hp = g2p(ref, language), g2p(hyp, language)
        if not rp:
            raise ValueError('No reference phonemes')
        alignment = edit_alignment(rp, hp)
        result.update(per=alignment['error_rate'], phoneme_alignment=alignment,
                      reference_phonemes=rp, transcript_phonemes=hp)
    except Exception as exc:
        result.update(per=None, per_error=str(exc))
    return result


def aggregate(rows, expected, errors=None):
    """Micro-average each language-specific metric; PER coverage is independent."""
    errors = errors or []
    result = {'available': bool(rows), 'n_expected': expected, 'n_evaluated': len(rows),
              'coverage': len(rows) / expected if expected else 0, 'per_sample': rows, 'errors': errors,
              'protocol': {'transcription': 'audio only; no target lyrics supplied to ASR',
                           'normalization': 'NFKC, casefold, remove []/<> annotations and punctuation',
                           'zh_per': 'pypinyin strict initials + tone3 finals (neutral tone 5)',
                           'en_per': 'phonemizer espeak en-us phones, no stress',
                           'aggregation': 'micro-average by reference units; zh CER and en WER separate',
                           'note': 'ASR/G2P errors can affect these scores; PER is not a timing metric'}}
    for key, align_key in (('cer', 'lexical_alignment'), ('wer', 'lexical_alignment'), ('per', 'phoneme_alignment')):
        eligible = [r for r in rows if key in r]
        valid = [r for r in eligible if r.get(key) is not None]
        units = sum(r[align_key]['n_reference'] for r in valid)
        errs = sum(sum(r[align_key][k] for k in ('substitutions', 'deletions', 'insertions')) for r in valid)
        result[key] = errs / units if units else None
        result[key + '_coverage'] = len(valid) / len(eligible) if eligible else None
    if any(r.get('per') is None for r in rows):
        result['available'] = False
        result['reason'] = 'PER failed for some samples; lexical scores/transcripts retained'
    return result
