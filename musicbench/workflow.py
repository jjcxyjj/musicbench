"""Task-defined evaluation policies and serial, YAML-backed experiments."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import yaml

from musicbench.evaluation.catalog import PRIMARY_RULES, SCALARS, resolve_plan

ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path):
    with open(path, encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f'Expected YAML mapping: {path}')
    return data


def save_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')


def save_json(path, data):
    from scripts.metric_worker import clean
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def choose(prompt, choices, multiple=False):
    if not choices:
        raise ValueError(f'{prompt}: no available choices')
    for i, name in enumerate(choices, 1):
        print(f'  {i}) {name}')
    while True:
        raw = input(f'{prompt} [1]' + (' (逗号分隔，可多选)' if multiple else '') + ': ').strip() or '1'
        try:
            selected = []
            for part in raw.split(','):
                part = part.strip()
                if part.isdigit():
                    index = int(part) - 1
                    if not 0 <= index < len(choices):
                        raise ValueError()
                    part = choices[index]
                if part not in choices:
                    raise ValueError()
                if part not in selected:
                    selected.append(part)
            if not multiple and len(selected) != 1:
                raise ValueError()
            return selected if multiple else selected[0]
        except (ValueError, IndexError):
            print('请输入有效选项。')


def ask(prompt, default=''):
    return input(f'{prompt} [{default}]: ').strip() or str(default)


def ask_bool(prompt, default=True):
    return choose(prompt, ['yes', 'no'] if default else ['no', 'yes']) == 'yes'


def ask_int(prompt, default, minimum=1):
    while True:
        try:
            value = int(ask(prompt, default))
            if value < minimum:
                raise ValueError()
            return value
        except ValueError:
            print(f'请输入不小于 {minimum} 的整数。')


def load_plugins(modules):
    for module in modules:
        importlib.import_module(module)


def interactive(catalog_path, task=None):
    catalog = read_yaml(catalog_path)
    load_plugins(catalog.get('plugins', []))
    profiles = catalog['tasks']
    task = task or choose('选择任务', list(profiles))
    if task not in profiles:
        raise ValueError(f'Unknown task: {task}')
    profile = profiles[task]
    rules = copy.deepcopy(profile['metrics'])
    dataset = dict(profile.get('dataset', {'name': 'manifest'}))
    dataset['name'] = choose('输入数据格式', ['manifest', 'feature_json']) if task == 'text2music' else dataset['name']
    key = 'root' if dataset['name'] == 'feature_json' else 'path'
    dataset[key] = ask('数据根目录' if key == 'root' else '输入清单 JSON 路径', dataset.get(key, 'dataset/manifest.json'))
    inputs = {}
    reference = None
    if task == 'text2music':
        inputs['style_prompt'] = ask_bool('清单中有风格描述吗？')
        inputs['lyrics'] = ask_bool('清单中有目标歌词吗？', False)
        inputs['reference_audio'] = ask_bool('有用于 FAD 的真实音乐参考集合吗？', False)
        if dataset['name'] == 'feature_json':
            dataset['caption_field'] = ask('风格描述字段名', 'Qwen_caption')
            dataset['lyrics_field'] = ask('歌词字段名', 'lyrics')
            dataset['language_field'] = ask('歌词语言字段名', 'language')
        if inputs['reference_audio']:
            source = choose('参考音频来源', ['directory', 'dataset'])
            reference = {'source': source}
            if source == 'directory':
                reference['path'] = ask('真实音乐参考目录')
                reference['num_samples'] = ask_int('参考抽样数（0 为全部）', 0, 0)
    from musicbench.algorithms import discover
    available = discover(task)
    names = choose('选择算法（示例算法只用于流程自测）', list(available), multiple=True)
    algorithms = []
    for name in names:
        item = dict(available[name])
        item['name'] = name
        item['env'] = ask(f'{name} conda 环境（current 使用主环境）', item.get('env', 'current'))
        item['cuda_visible_devices'] = ask(f'{name} 显卡编号（inherit 继承，cpu 禁用 CUDA）', item.get('cuda_visible_devices', 'inherit'))
        item['infer_script'] = ask(f'{name} 推理脚本', item.get('infer_script', 'infer.py'))
        params = dict(item.get('parameters', {}))
        for key, spec in item.get('parameter_schema', {}).items():
            default = params.get(key, spec.get('default', ''))
            params[key] = ask(f'{name}: {spec.get("label", key)} ({spec.get("type", "str")})', default)
        item['parameters'] = params
        if not item.get('parameter_schema'):
            import shlex
            item['args'] = shlex.split(ask(f'{name} 额外命令行参数', ''))
        algorithms.append(item)
    dataset['num_samples'] = ask_int('随机样本数', 10)
    config = {'version': 2, 'task': task, 'seed': ask_int('随机种子', 42, 0),
              'repetitions': ask_int('每条输入生成次数（全部评测，不挑最好结果）', 1),
              'plugins': catalog.get('plugins', []), 'dataset': dataset, 'inputs': inputs,
              'algorithms': algorithms, 'metric_rules': rules, 'require_metrics': True,
              'metric_options': {}, 'metric_environments': {}, 'ranking': {}}
    if reference:
        config['reference_audio'] = reference
    for name, rule in rules.items():
        if any(inputs.get(condition) is False for condition in rule.get('requires', [])):
            continue
        options = {}
        if name == 'songeval':
            options['repo'] = ask('已跑通的 SongEval 仓库路径', ROOT / 'third_party/SongEval')
            options['env'] = ask('SongEval conda 环境', 'songeval')
        else:
            config['metric_environments'][name] = ask(f'{name} conda 环境', ({'clap_score': 'clap', 'fad': 'clap'}.get(name, name)) if task == 'text2music' else 'current')
        options['cuda_visible_devices'] = ask(f'{name} 评测显卡编号（inherit/cpu）', 'inherit')
        options['device'] = ask(f'{name} 设备 auto/cpu/cuda', 'auto')
        if name in ('clap_score', 'fad', 'mulan', 'lyrics'):
            defaults = {'clap_score': 'laion/clap-htsat-fused', 'fad': 'laion/clap-htsat-fused',
                        'mulan': 'OpenMuQ/MuQ-MuLan-large', 'lyrics': 'large-v3'}
            options['model'] = ask(f'{name} 模型标识或本地目录', defaults[name])
        if name == 'audiobox':
            options['checkpoint'] = ask('Audiobox 权重路径（空白使用官方默认）', '')
        if name == 'lyrics':
            options['language'] = choose('清单未指定语言时的默认歌词语言', ['zh', 'en'])
            options['audio_source'] = choose('歌词音频处理协议（所有算法一致）', ['demucs', 'mix'])
            options['beam_size'] = ask_int('歌词识别 beam size', 5)
            if options['audio_source'] == 'demucs':
                options['separation'] = {'env': ask('人声分离 conda 环境', 'demucs'),
                                         'device': ask('人声分离设备 auto/cpu/cuda', 'auto'),
                                         'cuda_visible_devices': ask('人声分离显卡编号', options['cuda_visible_devices']),
                                         'model': ask('Demucs 模型', 'htdemucs')}
        config['metric_options'][name] = options
    return config


def environment_command(env, args, conda='conda'):
    if not env or env == 'current':
        return [sys.executable, *map(str, args)]
    flag = '-p' if os.path.isabs(env) else '-n'
    return [conda, 'run', '--no-capture-output', flag, env, 'python', *map(str, args)]


def resolve(path, base):
    path = Path(path).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def parameter_arguments(algo):
    args = []
    for key, value in algo.get('parameters', {}).items():
        spec = algo.get('parameter_schema', {}).get(key)
        if not spec:
            raise ValueError(f'{algo["name"]}: parameter {key!r} needs a parameter_schema entry')
        flag = spec.get('flag', '--' + key.replace('_', '-'))
        if spec.get('type') == 'bool':
            if value:
                args.append(flag)
        elif value is not None and value != '':
            args.extend([flag, str(value)])
    return args + algo.get('args', [])


def validate(config, base):
    import musicbench.tasks
    import musicbench.metrics
    import musicbench.datasets
    from musicbench.core.registry import get_task, get_metric, get_dataset
    config = copy.deepcopy(config)
    load_plugins(config.get('plugins', []))
    task = get_task(config['task'])
    if int(config.get('repetitions', 1)) <= 0:
        raise ValueError('repetitions must be positive')
    config['repetitions'] = int(config.get('repetitions', 1))
    config['seed'] = int(config.get('seed', 42))
    if 'metric_rules' not in config:
        if config['task'] == 'text2music':
            config['metric_rules'] = copy.deepcopy(PRIMARY_RULES)
            config.pop('metrics', None)
            config.pop('tier', None)
        else:
            allowed = config.get('allowed_metrics') or task.default_metrics()
            metrics = config.get('metrics') or task.default_metrics()
            if len(set(metrics)) != len(metrics):
                raise ValueError('Choose distinct metrics')
            for name in metrics:
                if name not in allowed:
                    raise ValueError(f'Metric {name} is incompatible with {config["task"]}')
            config['metric_rules'] = {m: {'category': 'task', 'requires': []} for m in metrics}
    if not config['metric_rules']:
        raise ValueError('Task must define evaluation metrics')
    for name, rule in config['metric_rules'].items():
        get_metric(name)
        if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError(f'Invalid metric name: {name}')
        if not isinstance(rule.get('requires', []), list):
            raise ValueError(f'{name}: requires must be a list')
    config['metrics'] = list(config['metric_rules'])
    for key, value in config.get('inputs', {}).items():
        if not isinstance(value, bool):
            raise ValueError(f'inputs.{key} must be true/false')
    dataset = config['dataset']
    get_dataset(dataset['name'])
    if int(dataset.get('num_samples', 10)) <= 0:
        raise ValueError('num_samples must be positive')
    for key in ('root', 'path'):
        if key in dataset:
            dataset[key] = resolve(dataset[key], base)
    ref = config.get('reference_audio', {})
    if ref and ref.get('source', 'directory') not in ('directory', 'dataset'):
        raise ValueError('reference_audio.source must be directory or dataset')
    if 'path' in ref:
        ref['path'] = resolve(ref['path'], base)
    if config.get('inputs', {}).get('reference_audio') is True and not ref:
        raise ValueError('Declared reference audio needs reference_audio.source/path')
    names = []
    for algo in config['algorithms']:
        name = algo['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Algorithm names must contain only letters, numbers, _ or -')
        names.append(name)
        if algo.get('task', config['task']) != config['task']:
            raise ValueError(f'{name}: incompatible task')
        algo['cwd'] = resolve(algo.get('cwd', '.'), base)
        algo['infer_script'] = resolve(algo.get('infer_script', 'infer.py'), Path(algo['cwd']))
        if not Path(algo['infer_script']).is_file():
            raise ValueError(f'Missing inference script: {algo["infer_script"]}')
        params = algo.setdefault('parameters', {})
        for key, spec in algo.get('parameter_schema', {}).items():
            value = params.get(key, spec.get('default'))
            kind = spec.get('type', 'str')
            if spec.get('flag') in ('--manifest', '--output'):
                raise ValueError('Algorithm parameter cannot override manifest/output')
            if value in ('', None):
                if spec.get('required'):
                    raise ValueError(f'{name}: parameter {key} is required')
                params[key] = value
                continue
            if kind == 'int':
                value = int(value)
            elif kind == 'float':
                import math
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f'{key} must be finite')
            elif kind == 'path':
                value = resolve(value, base)
            elif kind == 'bool':
                if not isinstance(value, bool):
                    if str(value).lower() not in ('true', 'false', 'yes', 'no', '1', '0'):
                        raise ValueError(f'Invalid bool parameter: {key}')
                    value = str(value).lower() in ('true', 'yes', '1')
            elif kind == 'str':
                value = str(value)
            else:
                raise ValueError(f'Unknown parameter type: {kind}')
            if 'choices' in spec and value not in spec['choices']:
                raise ValueError(f'{key} must be one of {spec["choices"]}')
            if 'minimum' in spec and value < spec['minimum']:
                raise ValueError(f'{key} must be >= {spec["minimum"]}')
            params[key] = value
        if not isinstance(algo.get('args', []), list) or not all(isinstance(v, str) for v in algo.get('args', [])):
            raise ValueError('Algorithm args must be a list of strings')
        if any(a in ('--manifest', '--output') or a.startswith(('--manifest=', '--output=')) for a in algo.get('args', [])):
            raise ValueError('Extra args cannot override manifest/output')
        parameter_arguments(algo)
    if not names or len(set(names)) != len(names):
        raise ValueError('Choose distinct, nonempty algorithms')
    for name, options in config.get('metric_options', {}).items():
        if name == 'songeval':
            options['repo'] = resolve(options.get('repo', ROOT / 'third_party/SongEval'), base)
        if options.get('checkpoint'):
            options['checkpoint'] = resolve(options['checkpoint'], base)
        if options.get('model') and (str(options['model']).startswith(('.', '/', '~'))):
            options['model'] = resolve(options['model'], base)
        if name == 'lyrics' and options.get('audio_source', 'demucs') not in ('demucs', 'mix'):
            raise ValueError('lyrics.audio_source must be demucs or mix')
    for key, spec in config.get('report_scalars', {}).items():
        if spec.get('direction') not in ('higher', 'lower', 'descriptive') or spec.get('category') not in ('aesthetics', 'lyrics', 'style', 'distribution'):
            raise ValueError(f'Invalid report scalar definition: {key}')
    return config


def prepare_samples(config, output):
    from scripts.package_dataset import package
    from scripts.score import load_manifest
    from musicbench.core.data import Sample
    ds = dict(config['dataset'])
    name, split = ds.pop('name'), ds.pop('split', 'test')
    ds['seed'] = config['seed']
    package(name, split, str(output / 'dataset'), **ds)
    manifest = output / 'dataset/manifest.json'
    samples = load_manifest(str(manifest))
    if not samples or len({s.id for s in samples}) != len(samples):
        raise ValueError('Dataset is empty or contains duplicate IDs')
    for sample in samples:
        # Existing captions remain usable as style descriptions, never as lyrics.
        if 'style_prompt' not in sample.metadata:
            sample.metadata['style_prompt'] = sample.metadata.get('caption', '')
        sample.metadata.setdefault('caption', sample.metadata.get('style_prompt', ''))
        if sample.audio_path and not Path(sample.audio_path).is_file():
            raise ValueError(f'Missing input/reference audio: {sample.audio_path}')
        if config['task'] != 'text2music' and not sample.audio_path:
            raise ValueError('This task requires input audio')
    ref = config.get('reference_audio', {})
    paths = []
    if config.get('inputs', {}).get('reference_audio') is not False and ref:
        if ref.get('source', 'directory') == 'dataset':
            paths = sorted({s.audio_path for s in samples if s.audio_path})
        else:
            directory = Path(ref['path'])
            if not directory.is_dir():
                raise ValueError(f'Reference directory missing: {directory}')
            extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
            paths = sorted(str(p.resolve()) for p in directory.rglob('*') if p.suffix.lower() in extensions and p.is_file())
            count = int(ref.get('num_samples', 0))
            if count < 0 or count > len(paths):
                raise ValueError('Invalid reference sample count')
            if count:
                paths = random.Random(config['seed']).sample(paths, count)
        if not paths:
            raise ValueError('Declared reference collection is empty')
    save_json(output / 'reference_manifest.json', [{'audio_path': path} for path in paths])
    repeated = []
    count = config['repetitions']
    for sample in samples:
        for rep in range(count):
            meta = {**sample.metadata, 'source_id': sample.id, 'repetition': rep,
                    'generation_seed': config['seed'] + rep}
            sid = sample.id if count == 1 else f'{sample.id}__rep_{rep:03d}'
            repeated.append(Sample(id=sid, audio_path=sample.audio_path,
                                   metadata=meta, ground_truth=sample.ground_truth))
    if len({s.id for s in repeated}) != len(repeated):
        raise ValueError('Repeated sample IDs collide')
    samples = repeated
    save_json(manifest, [{'id': s.id, 'audio_path': s.audio_path, 'metadata': s.metadata,
                          'ground_truth': s.ground_truth} for s in samples])
    return manifest, samples, paths


def execute(config, output):
    from scripts.score import load_predictions, validate_predictions
    from musicbench.core.ranking import rank, DEFAULT_WEIGHTS
    from musicbench.core.reproducibility import collect_env
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(config)
    save_yaml(output / 'config.yaml', config)
    save_yaml(output / 'status.yaml', {'status': 'running'})
    logs = output / 'logs'
    logs.mkdir()
    child_env = os.environ.copy()
    child_env.update(PYTHONPATH=str(ROOT) + os.pathsep + child_env.get('PYTHONPATH', ''),
                     MUSICBENCH_SEED=str(config['seed']), PYTHONDONTWRITEBYTECODE='1',
                     MPLCONFIGDIR=str(output / '.matplotlib'))
    conda = config.get('conda', os.environ.get('CONDA', 'conda'))
    child_env['CONDA'] = conda

    def run(label, env, args, cwd=ROOT, options=None):
        options = options or {}
        command = environment_command(env, args, conda)
        process_env = child_env.copy()
        process_env['PYTHONPATH'] = str(cwd) + os.pathsep + process_env.get('PYTHONPATH', '')
        gpu = str(options.get('cuda_visible_devices', 'inherit'))
        if options.get('device') == 'cpu':
            gpu = 'cpu'
        if gpu != 'inherit':
            process_env['CUDA_VISIBLE_DEVICES'] = '' if gpu == 'cpu' else gpu
        print(f'[run] {label} (env={env or "current"}, gpu={gpu})', flush=True)
        started = time.monotonic()
        with (logs / f'{label}.log').open('w') as log:
            log.write(json.dumps(command, ensure_ascii=False) + '\n')
            log.flush()
            proc = subprocess.Popen(command, cwd=cwd, env=process_env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = proc.wait(timeout=options.get('timeout', config.get('timeout', 3600)))
            except BaseException:
                # Kill the conda worker's process group as well; no GPU work leaks into the next stage.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return time.monotonic() - started

    def worker(label, metric, rows, options, extra=None):
        target = output / 'metric_results' / label / metric
        target.mkdir(parents=True, exist_ok=True)
        job = {'metric': metric, 'rows': rows, 'options': options,
               'artifact_dir': str(target / 'artifacts'), **(extra or {})}
        save_json(target / 'request.json', job)
        destination = target / 'result.json'
        try:
            env = options.get('env', config.get('metric_environments', {}).get(metric))
            run(f'score-{label}-{metric}', env, [ROOT / 'scripts/metric_worker.py',
                '--request', target / 'request.json', '--output', destination], options=options)
            result = json.loads(destination.read_text())
        except Exception as exc:
            result = {'available': False, 'coverage': 0.0, 'reason': f'{type(exc).__name__}: {exc}',
                      'log': str(logs / f'score-{label}-{metric}.log')}
        save_json(destination, result)
        return result

    try:
        manifest, samples, references = prepare_samples(config, output)
        plan = resolve_plan(config['metric_rules'], samples, config.get('inputs'), references)
        save_yaml(output / 'evaluation_plan.yaml', plan)
        required = {'beat_fmeasure': ('beats',), 'tempo_mae': ('tempo',),
                    'tag_map': ('tags',), 'tag_macro_f1': ('tags',), 'tag_macro_auc': ('tags',)}
        for metric in config['metrics']:
            for sample in samples:
                for field in required.get(metric, ()):
                    if field not in sample.ground_truth:
                        raise ValueError(f'{sample.id}: missing ground truth {field} for {metric}')
        scores, runtime, failures = {}, {}, []
        for algo in config['algorithms']:
            label = algo['name']
            pred = output / 'predictions' / label / 'predictions.jsonl'
            pred.parent.mkdir(parents=True, exist_ok=True)
            scores[label] = {}
            for name, info in plan.items():
                if info['status'] == 'not_applicable':
                    scores[label][name] = {'status': 'not_applicable', 'reason': info['reason']}
            try:
                seconds = run(f'infer-{label}', algo.get('env'), [algo['infer_script'], '--manifest', manifest,
                              '--output', pred, *parameter_arguments(algo)], algo['cwd'], algo)
                predictions = load_predictions(str(pred))
                validate_predictions(config['task'], predictions, samples)
                runtime[label] = {'inference_seconds': seconds, 'generated_count': len(predictions),
                                  'status': 'completed', 'note': 'Wall time includes environment launch and model loading'}
            except Exception as exc:
                if config['task'] != 'text2music':
                    raise
                failures.append(f'{label}: inference failed')
                runtime[label] = {'status': 'failed', 'reason': str(exc)}
                for name, info in plan.items():
                    if info['status'] == 'scheduled':
                        scores[label][name] = {'available': False, 'coverage': 0.0,
                                              'reason': 'Inference failed; see inference log'}
                save_json(output / 'partial_scores.json', scores)
                continue
            by_id = {p.id: p for p in predictions}
            for metric, info in plan.items():
                if info['status'] != 'scheduled':
                    continue
                ids = set(info['eligible_ids'])
                rows = [{'id': s.id, 'audio_path': by_id[s.id].output['audio_path'],
                         'metadata': s.metadata} for s in samples if s.id in ids] if config['task'] == 'text2music' else []
                options = dict(config.get('metric_options', {}).get(metric, {}))
                if metric in ('audiobox', 'lyrics', 'mulan', 'clap_score', 'fad'):
                    if metric == 'lyrics' and options.get('audio_source', 'demucs') == 'demucs':
                        separated = worker(label, 'separation', rows, options.get('separation', {}))
                        if not separated.get('available'):
                            result = {'available': False, 'coverage': 0.0,
                                      'reason': 'Vocal separation failed', 'errors': separated.get('errors', separated.get('reason'))}
                            save_json(output / 'metric_results' / label / metric / 'result.json', result)
                            scores[label][metric] = result
                            continue
                        vocals = {r['id']: r['audio_path'] for r in separated['per_sample']}
                        rows = [{**r, 'audio_path': vocals[r['id']]} for r in rows]
                    scores[label][metric] = worker(label, metric, rows, options, {'reference_paths': references})
                elif metric == 'songeval':
                    target = output / 'metric_results' / label / metric
                    target.mkdir(parents=True, exist_ok=True)
                    save_json(target / 'input.json', [{'id': r['id'], 'audio_path': r['audio_path'],
                              'has_vocal': r['metadata'].get('has_vocal')} for r in rows])
                    try:
                        run(f'score-{label}-{metric}', options.get('env', config.get('metric_environments', {}).get(metric)), [ROOT / 'scripts/songeval_worker.py',
                            '--repo', options.get('repo', ROOT / 'third_party/SongEval'), '--input', target / 'input.json',
                            '--output', target / 'raw.json', '--device', options.get('device', 'auto')], options=options)
                        from musicbench.evaluation.primary import aggregate
                        raw = json.loads((target / 'raw.json').read_text())
                        from musicbench.metrics.songeval import DIMENSIONS
                        result = aggregate(raw['per_sample'], DIMENSIONS, len(rows), raw['errors'])
                        result['model'] = raw['model']
                        eligible_vocals = [r for r in rows if r['metadata'].get('has_vocal') is True]
                        result['naturalness_n_expected'] = len(eligible_vocals)
                    except Exception as exc:
                        result = {'available': False, 'coverage': 0.0, 'reason': str(exc)}
                    scores[label][metric] = result
                    save_json(target / 'result.json', result)
                else:
                    # Existing registry metrics/plugins remain available for other tasks and extensions.
                    target = output / 'metric_results' / label / metric
                    args = [ROOT / 'scripts/score.py', '--task', config['task'], '--manifest', manifest,
                            '--predictions', pred, '--labels', label, '--metrics', metric, '--output', target]
                    if config.get('plugins'):
                        args += ['--plugins', *config['plugins']]
                    try:
                        run(f'score-{label}-{metric}', config.get('metric_environments', {}).get(metric), args, options=options)
                        scores[label].update(json.loads((target / 'scores.json').read_text())['algorithms'][label])
                    except Exception as exc:
                        scores[label][metric] = {'available': False, 'coverage': 0.0, 'reason': str(exc)}
                save_json(output / 'partial_scores.json', scores)
            # Preserve a completed per-algorithm report before starting another model.
            save_json(pred.parent / 'metrics.json', scores[label])
        incomplete = [f'{label}/{metric}' for label, values in scores.items() for metric, value in values.items()
                      if value.get('available') is False or value.get('coverage', 1) < 1]
        weights = dict(DEFAULT_WEIGHTS)
        weights.update({k: (v['weight'], v['direction'] == 'higher') for k, v in config.get('ranking', {}).items()})
        is_generation = config['task'] == 'text2music'
        data = {'task': config['task'], 'metrics': config['metrics'], 'evaluation_plan': plan,
                'metric_definitions': {**SCALARS, **config.get('report_scalars', {})} if is_generation else {}, 'sample_count': len(samples),
                'seed': config['seed'], 'repetitions': config['repetitions'], 'runtime': runtime,
                'algorithms': scores, 'ranking': [] if is_generation else rank(scores, weights),
                'ranking_weights': weights, 'incomplete': incomplete,
                'ranking_note': 'Separate automatic metrics; no aggregate quality score. PC describes complexity, not superiority.'
                if is_generation else 'Relative min-max comparison only, not absolute quality.'}
        results = output / 'results'
        save_json(results / 'scores.json', data)
        save_json(results / 'env.json', collect_env(config['seed']))
        run('report', 'current', [ROOT / 'scripts/plot.py', '--scores', results / 'scores.json', '--output', results])
        if failures or (incomplete and config.get('require_metrics', True)):
            raise RuntimeError('Incomplete experiment (reports retained): ' + ', '.join(failures + incomplete))
        save_yaml(output / 'status.yaml', {'status': 'completed'})
    except BaseException as exc:
        save_yaml(output / 'status.yaml', {'status': 'failed', 'reason': str(exc)})
        raise
    print(f'Report: {output / "results/report.md"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Task-defined serial MusicBench experiment')
    parser.add_argument('task', nargs='?')
    parser.add_argument('--catalog', default=str(ROOT / 'configs/tasks.yaml'))
    parser.add_argument('--config')
    parser.add_argument('--save-config')
    parser.add_argument('--output')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.config:
            path = Path(args.config).resolve()
            config, base = read_yaml(path), path.parent
        else:
            config, base = interactive(args.catalog, args.task), ROOT
        config = validate(config, base)
        if not args.config or args.save_config:
            saved = Path(args.save_config or ROOT / 'configs/experiment.local.yaml').resolve()
            save_yaml(saved, config)
            print(f'YAML: {saved}')
        print('Task metric policy: ' + ', '.join(config['metrics']))
        if not args.dry_run:
            execute(config, args.output or ROOT / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'[error] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
