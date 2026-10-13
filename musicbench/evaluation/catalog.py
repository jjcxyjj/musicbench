"""Task policies and scalar report definitions; no model imports."""
PRIMARY_RULES = {
    'audiobox': {'category': 'aesthetics', 'requires': []},
    'songeval': {'category': 'aesthetics', 'requires': []},
    'lyrics': {'category': 'lyrics', 'requires': ['lyrics']},
    'clap_score': {'category': 'style', 'requires': ['style_prompt']},
    'mulan': {'category': 'style', 'requires': ['style_prompt']},
    'fad': {'category': 'distribution', 'requires': ['reference_audio']},
}
CATEGORIES = {'aesthetics': '美学评估（含结构）', 'lyrics': '歌词匹配度',
              'style': '风格匹配度', 'distribution': '与真实音乐分布的距离'}
SCALARS = {
    **{f'audiobox.{key}': {'category': 'aesthetics', 'direction': 'descriptive' if key == 'PC' else 'higher'}
       for key in ('CE', 'CU', 'PC', 'PQ')},
    **{f'songeval.{key}': {'category': 'aesthetics', 'direction': 'higher'}
       for key in ('coherence', 'musicality', 'memorability', 'clarity', 'naturalness')},
    **{f'lyrics.{key}': {'category': 'lyrics', 'direction': 'lower'} for key in ('per', 'cer', 'wer')},
    'clap_score.clap': {'category': 'style', 'direction': 'higher'},
    'mulan.similarity': {'category': 'style', 'direction': 'higher'},
    'fad.fad': {'category': 'distribution', 'direction': 'lower'},
}


def resolve_plan(rules, samples, inputs=None, reference_paths=None):
    """Missing optional inputs are N/A; declared inputs missing data are errors."""
    inputs = inputs or {}
    reference_paths = reference_paths or []
    plan = {}
    for name, rule in rules.items():
        eligible = list(samples)
        reason = None
        for condition in rule.get('requires', []):
            if condition == 'reference_audio':
                if inputs.get(condition) is False or not reference_paths:
                    eligible = []
                    reason = '未提供真实音乐参考集合'
            else:
                if inputs.get(condition) is False:
                    eligible = []
                    reason = f'未提供 {condition}'
                else:
                    eligible = [s for s in eligible if str(s.metadata.get(condition) or '').strip()]
                    if inputs.get(condition) is True and len(eligible) != len(samples):
                        raise ValueError(f'Declared {condition} is missing for one or more samples')
                    if not eligible:
                        reason = f'未提供 {condition}'
        plan[name] = {'category': rule.get('category', 'other'),
                      'status': 'scheduled' if eligible else 'not_applicable',
                      'eligible_ids': [s.id for s in eligible], 'reason': reason}
    return plan
