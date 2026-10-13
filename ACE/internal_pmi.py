"""Existing WAV + target lyrics -> ACE LM normalized PMI (not ASR accuracy)."""
from pathlib import Path
import os
import math
import sys
from shared import device_name, local_directory, summarize


def evaluate(rows, options, references=None):
    repo = Path(options['repo']).resolve()
    checkpoints = repo / 'checkpoints'
    dit_name = options.get('dit_model', 'acestep-v15-turbo')
    lm_name = options.get('lm_model', 'acestep-5Hz-lm-0.6B')
    # Do not enter ACE's automatic downloader when assets are absent.
    for name in (dit_name, lm_name, 'vae', 'Qwen3-Embedding-0.6B'):
        local_directory(checkpoints / name, 'ACE ' + name)
    sys.path.insert(0, str(repo))
    os.environ['ACESTEP_CHECKPOINTS_DIR'] = str(checkpoints)
    from acestep.model_downloader import check_main_model_exists
    if not check_main_model_exists(checkpoints):
        raise ValueError(f'ACE 公共模型文件不完整：{checkpoints}；请手工补齐，程序不下载。')
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.core.scoring.lm_score import calculate_pmi_score_per_condition
    device = device_name(options.get('device', 'auto'))
    dit, lm = AceStepHandler(), LLMHandler()
    message, success = dit.initialize_service(project_root=str(repo), config_path=dit_name,
                                              device=device, use_flash_attention=False,
                                              compile_model=False, offload_to_cpu=True)
    if not success:
        raise RuntimeError(message)
    message, success = lm.initialize(checkpoint_dir=str(checkpoints), lm_model_path=lm_name,
                                     backend='pt', device=device, offload_to_cpu=True)
    if not success:
        raise RuntimeError(message)
    results, errors = [], []
    for row in rows:
        try:
            # Every algorithm is judged from its final WAV, never its planned codes.
            codes = dit.convert_src_audio_to_codes(row['audio_path'])
            if not isinstance(codes, str) or not codes.startswith('<|audio_code_'):
                raise ValueError(f'ACE audio encoding failed: {codes}')
            scores, _, message = calculate_pmi_score_per_condition(
                lm_handler=lm, audio_codes=codes, lyrics=row['metadata']['lyrics'],
                metadata={'caption': None}, topk=int(options.get('topk', 10)),
                score_scale=float(options.get('score_scale', 0.1)))
            if 'lyrics' not in scores:
                raise RuntimeError(message)
            value = float(scores['lyrics'])
            if not math.isfinite(value):
                raise ValueError('Non-finite ACE lyric PMI')
            results.append({'id': row['id'], 'lyrics_pmi_normalized': value})
        except Exception as exc:
            errors.append({'id': row['id'], 'reason': str(exc)})
    result = summarize(results, ['lyrics_pmi_normalized'], len(rows), errors)
    result['model'] = {'repo': str(repo), 'dit_model': dit_name, 'lm_model': lm_name,
                      'scope': 'ACE LM 对目标歌词的条件支持；不是 PER/CER/WER，也不是 DiT 注意力对齐分数。'}
    return result
