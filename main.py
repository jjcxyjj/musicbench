"""Evaluate existing generated audio in one already activated environment."""
import argparse
from datetime import datetime
import gc
import importlib.util
import json
from pathlib import Path
import sys

import yaml
from shared import enforce_offline

ROOT = Path(__file__).resolve().parent
SCORERS = {'songeval': 'songeval', 'clap': 'CLAP', 'audiobox': 'audiobox-aesthetics',
           'muq_mulan': 'MuQ', 'fad': 'FAD', 'per': 'PER', 'ace_captioner': 'ACE', 'ace_transcriber': 'ACE'}


def ask_yes_no(question, default):
    while True:
        answer = input(f'{question} [{"Y/n" if default else "y/N"}]: ').strip().lower()
        if not answer:
            return default
        if answer in ('y', 'yes', '是', '有', '1'):
            return True
        if answer in ('n', 'no', '否', '没有', '无', '0'):
            return False
        print('请输入 y 或 n。')


def resolve_path(value, base):
    return str((base / Path(value).expanduser()).resolve())


def load_input(path):
    """JSON contains generated audio paths and optional annotations, not algorithms."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    records = data if isinstance(data, list) else data.get('samples') if isinstance(data, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError('JSON 必须是非空记录数组，或包含 samples 数组的对象。')
    samples = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get('metadata', {}), dict):
            raise ValueError(f'第 {index + 1} 条记录及 metadata 必须是对象。')
        audio = record.get('audio_path') or record.get('generated_audio_path')
        if not audio:
            raise ValueError(f'第 {index + 1} 条缺 audio_path：请明确填写算法输出音频，不能直接把原始数据的 path 当生成结果。')
        audio = resolve_path(audio, path.parent)
        if not Path(audio).is_file():
            raise ValueError(f'算法输出音频不存在：{audio}')
        metadata = {**record, **record.get('metadata', {})}
        metadata['style_prompt'] = metadata.get('style_prompt') or metadata.get('Qwen_caption') or metadata.get('caption') or ''
        metadata['lyrics'] = metadata.get('lyrics') or ''
        if not isinstance(metadata['style_prompt'], str) or not isinstance(metadata['lyrics'], str):
            raise ValueError('风格描述和歌词必须为字符串。')
        if metadata.get('has_vocal') is not None and not isinstance(metadata['has_vocal'], bool):
            raise ValueError('has_vocal 必须为 JSON true/false，不能是字符串。')
        gt = metadata.get('gt_audio_path') or metadata.get('path_original')
        samples.append({'id': str(record.get('id', index)), 'audio_path': audio,
                        'algorithm': str(record.get('algorithm', 'generated')), 'metadata': metadata,
                        'gt_audio_path': resolve_path(gt, path.parent) if gt else ''})
    keys = [(s['algorithm'], s['id']) for s in samples]
    if len(set(keys)) != len(keys):
        raise ValueError('同一算法内的样本 ID 必须唯一。')
    references = data.get('reference_audio', []) if isinstance(data, dict) else []
    if not isinstance(references, list) or not all(isinstance(p, str) for p in references):
        raise ValueError('reference_audio 必须是音频路径数组。')
    references = [resolve_path(p, path.parent) for p in references]
    references += [s['gt_audio_path'] for s in samples if s['gt_audio_path']]
    return samples, sorted(set(references))


def choose_inputs(samples, references, interactive=True):
    flags = {'lyrics': any(s['metadata']['lyrics'].strip() and s['metadata'].get('has_vocal') is True for s in samples),
             'gt_audio': bool(references), 'style_prompt': any(s['metadata']['style_prompt'].strip() for s in samples)}
    if interactive:
        flags['lyrics'] = ask_yes_no('有目标歌词吗？（纯音乐无需歌词）', flags['lyrics'])
        flags['gt_audio'] = ask_yes_no('有 GT / 真实参考音频吗？（用于 FAD）', flags['gt_audio'])
        if flags['gt_audio'] and not references:
            raw_directory = input('GT 音频目录：').strip()
            if not raw_directory:
                raise ValueError('GT 音频目录不能为空。')
            directory = Path(raw_directory).expanduser().resolve()
            if not directory.is_dir():
                raise ValueError(f'GT 目录不存在：{directory}')
            references = sorted(str(p) for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in ('.wav', '.flac', '.mp3', '.ogg', '.m4a'))
        flags['style_prompt'] = ask_yes_no('有风格描述吗？', flags['style_prompt'])
    if flags['gt_audio']:
        if not references:
            raise ValueError('选择了有 GT，但未找到参考音频。')
        for reference in references:
            if not Path(reference).is_file():
                raise ValueError(f'GT 音频不存在：{reference}')
    else:
        references = []
    return flags, references


def select_rows(metric, samples, flags, references):
    if metric in ('songeval', 'audiobox', 'ace_captioner'):
        return samples, None
    if metric in ('muq_mulan', 'clap'):
        rows = [s for s in samples if flags['style_prompt'] and s['metadata']['style_prompt'].strip()]
        return rows, None if rows else '没有风格描述'
    if metric in ('ace_transcriber', 'per'):
        rows = [s for s in samples if flags['lyrics'] and s['metadata'].get('has_vocal') is True and s['metadata']['lyrics'].strip()]
        return rows, None if rows else '没有明确标注有人声且有目标歌词的样本'
    return (samples, None) if flags['gt_audio'] and references else ([], '没有 GT / 真实参考音频')


def load_scorer(name):
    path = ROOT / SCORERS[name] / 'score.py'
    spec = importlib.util.spec_from_file_location('score_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if name.startswith('ace_'):
        return lambda rows, options, references: module.evaluate(rows, {**options, 'task': name.removeprefix('ace_')}, references)
    return module.evaluate


def run(samples, references, flags, config, output, metrics=None, plan_only=False):
    enabled = metrics if metrics is not None else config.get('enabled_metrics', ['songeval', 'audiobox'])
    if not enabled or any(name not in SCORERS for name in enabled):
        raise ValueError('enabled_metrics 必须从 ' + ', '.join(SCORERS) + ' 中选择。')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups = list(dict.fromkeys(s['algorithm'] for s in samples))
    result = {'inputs': flags, 'reference_audio': references, 'algorithms': {}, 'enabled_metrics': enabled,
              'note': '既有音频的自动评分；不生成音乐、不计算综合质量总分。ACE captioner 输出描述，transcriber 识别后计算歌词错误率；FAD 编码器见模型信息。'}
    (output / 'input.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    if not plan_only:
        enforce_offline()
    for group in groups:
        group_rows = [s for s in samples if s['algorithm'] == group]
        result['algorithms'][group] = {}
        for name in SCORERS:
            rows, reason = select_rows(name, group_rows, flags, references)
            status = 'disabled' if name not in enabled else 'not_applicable' if not rows else 'scheduled'
            entry = {'status': status, 'eligible_ids': [s['id'] for s in rows],
                     'excluded_ids': [s['id'] for s in group_rows if s not in rows], 'reason': reason}
            print(f'[{status}] {group} / {name}: {len(rows)} 条')
            if status == 'scheduled' and not plan_only:
                options = {**config.get(name, {}), 'device': config.get(name, {}).get('device', config.get('device', 'auto'))}
                try:
                    score = load_scorer(name)(rows, options, references)
                    entry.update(score)
                except Exception as exc:
                    entry.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
                    print('  失败：' + entry['reason'])
                finally:
                    gc.collect()
                    torch = sys.modules.get('torch')
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            result['algorithms'][group][name] = entry
            (output / 'scores.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if not plan_only:
        from reporting.write import write_report
        try:
            write_report(result, output)
        except Exception as exc:
            result['report_error'] = f'{type(exc).__name__}: {exc}'
            print('报告生成失败（分数已保留）：' + result['report_error'])
    failed = bool(result.get('report_error')) or any(v['status'] in ('failed', 'partial') for scores in result['algorithms'].values() for v in scores.values())
    result['status'] = 'planned' if plan_only else 'failed' if failed else 'completed'
    (output / 'scores.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'结果：{output}')
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='在已激活的 songeval 环境中评估已有生成音频')
    parser.add_argument('--input', help='算法输出音频和标注 JSON')
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--output')
    parser.add_argument('--metrics', nargs='+', choices=list(SCORERS))
    parser.add_argument('--no-interactive', action='store_true', help='按 JSON 字段自动判断可用输入')
    parser.add_argument('--plan-only', action='store_true', help='只检查输入和评测计划，不加载模型')
    args = parser.parse_args(argv)
    try:
        input_path = args.input or input('算法输出音频 JSON 路径：').strip()
        config_path = Path(args.config).resolve()
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if not isinstance(config, dict):
            raise ValueError('config.yaml 必须为配置映射。')
        for name in SCORERS:
            for key, value in config.get(name, {}).items():
                if key in ('repo', 'model', 'checkpoint', 'muq_model', 'audio_model', 'text_model', 'espeak_library') and value:
                    config[name][key] = resolve_path(value, config_path.parent)
        samples, references = load_input(input_path)
        flags, references = choose_inputs(samples, references, not args.no_interactive)
        return run(samples, references, flags, config,
                   args.output or ROOT / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
                   args.metrics, args.plan_only)
    except (OSError, ValueError, EOFError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
