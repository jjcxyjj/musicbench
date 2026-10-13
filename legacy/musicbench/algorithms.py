"""Register/discover independent model adapters without editing harness code."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETERS = {
    'checkpoint': {'flag': '--checkpoint', 'type': 'path', 'label': '模型权重路径', 'default': ''},
    'duration': {'flag': '--duration', 'type': 'float', 'minimum': 0.1, 'label': '生成时长（秒）', 'default': 30},
    'steps': {'flag': '--steps', 'type': 'int', 'minimum': 1, 'label': '采样步数', 'default': 30},
    'guidance': {'flag': '--guidance', 'type': 'float', 'label': '引导强度', 'default': 3},
    'sr': {'flag': '--sr', 'type': 'int', 'minimum': 8000, 'label': '采样率', 'default': 48000},
}


def discover(task=None, directory=None):
    directory = Path(directory or ROOT / 'algorithms')
    found = {}
    for path in sorted(directory.glob('*/config.yaml')):
        item = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(item, dict) or not item.get('task'):
            raise ValueError(f'Algorithm config needs a task: {path}')
        if task and item['task'] != task:
            continue
        if item.get('name', path.parent.name) != path.parent.name:
            raise ValueError(f'Algorithm name must match its directory: {path}')
        item = dict(item, name=path.parent.name)
        cwd = Path(item.get('cwd') or '.').expanduser()
        if not cwd.is_absolute():
            cwd = path.parent / cwd
        item['cwd'] = str(cwd.resolve())
        script = Path(item.get('infer_script') or 'infer.py').expanduser()
        if not script.is_absolute():
            script = cwd / script
        item['infer_script'] = str(script.resolve())
        found[item['name']] = item
    return found


def check_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError('Parameter schema must be a YAML mapping')
    for name, spec in schema.items():
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
            raise ValueError(f'Invalid parameter name: {name!r}')
        if not isinstance(spec, dict) or spec.get('type', 'str') not in ('str', 'int', 'float', 'path', 'bool'):
            raise ValueError(f'Invalid parameter definition: {name}')
        flag = spec.get('flag', '--' + name.replace('_', '-'))
        if not isinstance(flag, str) or not re.fullmatch(r'--[A-Za-z][A-Za-z0-9_-]*', flag):
            raise ValueError(f'Invalid parameter flag: {flag!r}')
        if flag in ('--manifest', '--output'):
            raise ValueError('manifest/output flags are reserved by the harness')
    return schema


def register(name, task, env, script=None, cwd=None, schema=None, gpu='inherit', directory=None):
    """Create a config and optional adapter stub. Existing registrations are preserved."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name or ''):
        raise ValueError('Algorithm name must use letters, digits, _ or -, starting with a letter/digit')
    if not task or not env:
        raise ValueError('task and env are required')
    destination = Path(directory or ROOT / 'algorithms') / name
    if destination.exists():
        raise FileExistsError(f'Algorithm already exists; edit its config explicitly: {destination}')
    scaffold = script is None
    if scaffold and task != 'text2music':
        raise ValueError('Scaffold supports text2music; provide --script for other tasks')
    workdir = Path(cwd).expanduser().resolve() if cwd else destination.resolve()
    if not scaffold:
        script_path = Path(script).expanduser()
        if not script_path.is_absolute():
            script_path = (workdir if cwd else Path.cwd()) / script_path
        script_path = script_path.resolve()
        if not script_path.is_file():
            raise ValueError(f'Inference adapter does not exist: {script_path}')
        if cwd and not workdir.is_dir():
            raise ValueError(f'Algorithm cwd does not exist: {workdir}')
    elif cwd:
        raise ValueError('--cwd requires --script; scaffold adapters run in their new algorithm directory')
    parameters = check_schema(copy.deepcopy(DEFAULT_PARAMETERS if scaffold and schema is None else schema if schema is not None else {}))
    config = {'name': name, 'task': task, 'env': env, 'cuda_visible_devices': str(gpu),
              'infer_script': 'infer.py' if scaffold else str(script_path),
              'parameter_schema': parameters, 'parameters': {}, 'args': []}
    if not scaffold:
        config['cwd'] = str(workdir if cwd else script_path.parent)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        if scaffold:
            shutil.copyfile(Path(__file__).parent / 'templates/algorithm_infer.py.txt', destination / 'infer.py')
        (destination / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
        (destination / 'README.md').write_text(
            '# ' + name + '\n\n' +
            ('在 infer.py 的 load_model()、generate() 中接入真实模型；接入前模板会明确报错。\n\n'
             if scaffold else '已登记外部推理适配脚本；确认它接受 --manifest、--output 并写出规定 JSONL。\n\n') +
            '参数在 config.yaml 的 parameter_schema 声明，主交互会逐项询问。环境和权重需自行准备。\n'
            '相对 cwd 以 config.yaml 所在目录解析，相对 infer_script 以 cwd 解析。\n', encoding='utf-8')
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination


def main(argv=None):
    from .workflow import ask, choose, read_yaml
    p = argparse.ArgumentParser(description='Register or list MusicBench algorithm adapters')
    commands = p.add_subparsers(dest='command', required=True)
    add = commands.add_parser('add', help='Register an adapter or create a model adapter template')
    for key in ('name', 'task', 'env', 'gpu', 'script', 'cwd', 'schema'):
        add.add_argument('--' + key)
    listing = commands.add_parser('list', help='List registered adapters')
    listing.add_argument('--task')
    args = p.parse_args(argv)
    try:
        if args.command == 'list':
            for name, item in discover(args.task).items():
                print(f"{name}: task={item['task']} env={item.get('env', 'current')} cwd={item['cwd']} script={item['infer_script']}")
            return 0
        interactive = not all((args.name, args.task, args.env))
        name = args.name or ask('新算法名称')
        tasks = read_yaml(ROOT / 'configs/tasks.yaml')['tasks']
        task = args.task or choose('算法支持的任务', list(tasks))
        if task not in tasks:
            raise ValueError(f'Add the new task to configs/tasks.yaml first: {task}')
        env = args.env or ask('已有 conda 环境名（current 使用主环境）', name)
        gpu = args.gpu or (ask('显卡编号（inherit/cpu）', 'inherit') if interactive else 'inherit')
        script, cwd = args.script, args.cwd
        if interactive and script is None:
            script = ask('已有推理适配脚本绝对路径（留空创建模板）') or None
            if script and cwd is None:
                cwd = ask('算法工作目录', str(Path(script).expanduser().resolve().parent))
        schema = read_yaml(args.schema) if args.schema else None
        result = register(name, task, env, script, cwd, schema, gpu)
        print(f'Algorithm registered: {result}')
        if script is None:
            print('Template created. Implement load_model() and generate() before running an experiment.')
        print('Run ./run.sh; the algorithm will appear automatically under its task.')
        return 0
    except (ValueError, OSError) as exc:
        print(f'[error] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
