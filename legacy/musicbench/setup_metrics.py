"""Offline evaluator diagnostics; installation and weight acquisition are manual."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('audiobox', 'songeval', 'lyrics', 'clap_score', 'mulan', 'fad')
COMPONENTS = {'audiobox': 'metric-audiobox.yml', 'songeval': 'songeval.yml',
              'clap': 'metric-clap.yml', 'mulan': 'metric-mulan.yml',
              'lyrics': 'metric-lyrics.yml', 'demucs': 'metric-demucs.yml'}
SOURCES = {'audiobox': 'https://github.com/facebookresearch/audiobox-aesthetics',
           'songeval': 'https://github.com/ASLP-lab/SongEval',
           'clap': 'https://github.com/huggingface/transformers',
           'mulan': 'https://github.com/tencent-ailab/MuQ',
           'lyrics': 'https://github.com/SYSTRAN/faster-whisper',
           'demucs': 'https://github.com/facebookresearch/demucs'}


def merge_defaults(defaults, overrides):
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        result[key] = merge_defaults(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def components_for(metrics, separation=True):
    result = []
    for metric in metrics:
        if metric not in METRICS:
            raise ValueError(f'Unknown metric: {metric}')
        component = 'clap' if metric in ('clap_score', 'fad') else metric
        if component not in result:
            result.append(component)
        if component == 'lyrics' and separation and 'demucs' not in result:
            result.append('demucs')
    return result


def parse_reuse(values):
    result = {}
    for value in values:
        key, sep, env = value.partition('=')
        if not sep or key not in COMPONENTS or not env.strip():
            raise ValueError('--reuse requires COMPONENT=existing_env_name_or_absolute_path')
        result[key] = env.strip()
    return result


def env_args(conda, env):
    return [conda, 'run', '--no-capture-output', '-p' if os.path.isabs(env) else '-n', env]


def run_setup(args):
    profile_path = Path(args.profile).expanduser().resolve()
    if args.init_config:
        if args.dry_run:
            print(f'Would copy example to {profile_path}; no changes made.')
            return 0
        if profile_path.exists():
            raise ValueError(f'Profile already exists, edit it manually: {profile_path}')
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        if args.shared_env:
            data = yaml.safe_load((ROOT / 'configs/metrics.offline.example.yaml').read_text())
            data['evaluation_environment'] = args.shared_env
            profile_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        else:
            shutil.copyfile(ROOT / 'configs/metrics.offline.example.yaml', profile_path)
        print(f'Example copied: {profile_path}; edit environment names and local paths before checking.')
        return 0
    if not profile_path.is_file():
        print(f'Profile missing: {profile_path}\nCopy configs/metrics.offline.example.yaml or run --init-config, then edit local paths.\nSee docs/METRIC_SETUP.md')
        return 0 if args.dry_run else 1
    from musicbench.workflow import load_metric_profile, use_shared_environment, evaluation_environment
    profile = load_metric_profile(profile_path)
    if args.shared_env:
        profile = use_shared_environment(profile, args.shared_env)
    overrides = parse_reuse(args.reuse)
    lyrics = profile.get('metric_options', {}).get('lyrics', {})
    separation = not args.no_separation and lyrics.get('audio_source', 'demucs') == 'demucs'
    components = components_for(args.metrics, separation)
    jobs = []
    for component in components:
        metric = ('clap_score' if 'clap_score' in args.metrics else 'fad') if component == 'clap' else component
        options = dict(lyrics.get('separation', {})) if component == 'demucs' else dict(profile.get('metric_options', {}).get(metric, {}))
        env = overrides.get(component) or evaluation_environment(profile, metric, options)
        jobs.append((component, env, options))
        print(f'{component}: env={env or "NOT CONFIGURED"}; local assets={options}')
    if args.dry_run:
        print('Plan only; no commands executed. No installation or downloading is performed in any mode.')
        return 0
    conda = args.conda
    paths, names = set(), set()
    if any(env != 'current' for _, env, _ in jobs):
        if not (Path(conda).is_file() or shutil.which(conda)):
            raise ValueError('conda executable missing; specify --conda /absolute/path/to/conda. See docs/METRIC_SETUP.md')
        info = subprocess.run([conda, 'env', 'list', '--json'], check=True, capture_output=True, text=True,
                              env={**os.environ, 'CONDA_OFFLINE': 'true'})
        paths = {str(Path(p).resolve()) for p in json.loads(info.stdout)['envs']}
        names = {Path(p).name for p in paths}
    logs = ROOT / 'work/metric-check' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logs.mkdir(parents=True, exist_ok=False)
    process_env = {**os.environ, **profile.get('runtime_environment', {}),
                   'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    report = {'profile': str(profile_path), 'offline': True, 'mode': 'imports' if args.libraries_only else 'check', 'components': {}}
    for component, env, options in jobs:
        result = {'status': 'failed'}
        try:
            if env != 'current' and (not env or not (str(Path(env).resolve()) in paths if os.path.isabs(env) else env in names)):
                template = 'evaluation.yml' if env == profile.get('evaluation_environment') else COMPONENTS[component]
                raise ValueError(f'Conda environment missing: {env!r}. Prepare envs/{template} on a connected machine and transfer it; see docs/METRIC_SETUP.md')
            request = logs / f'{component}-request.json'
            destination = logs / f'{component}-result.json'
            request.write_text(json.dumps({'component': component, 'mode': report['mode'], 'options': options}))
            with (logs / f'{component}.log').open('w') as handle:
                python_command = [sys.executable] if env == 'current' else [*env_args(conda, env), 'python']
                subprocess.run([*python_command, str(ROOT / 'scripts/prepare_metric.py'),
                                '--request', str(request), '--output', str(destination)], env=process_env,
                               stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
            result = json.loads(destination.read_text())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result['reason'] = str(exc)
        result.update(env=env, source=SOURCES[component], tutorial='docs/METRIC_SETUP.md')
        report['components'][component] = result
        print(f'[{result["status"]}] {component}: {result.get("reason", "local checks passed")}')
        if result.get('missing_dependencies'):
            print('  Missing/broken libraries: ' + ', '.join(result['missing_dependencies']))
        if result['status'] != 'ready':
            print(f'  Code: {SOURCES[component]} ; manual setup: docs/METRIC_SETUP.md')
    report['ready'] = all(v['status'] == 'ready' for v in report['components'].values())
    (logs / 'status.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'Diagnostics: {logs}; profile unchanged. No environment, repository or weights installed/downloaded.')
    return 0 if report['ready'] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description='Offline checks only; manually install evaluator libraries and local weights')
    p.add_argument('--profile', '--output', default=str(ROOT / 'configs/metrics.local.yaml'))
    p.add_argument('--init-config', action='store_true', help='Copy local-path example without overwriting existing configuration')
    p.add_argument('--metrics', nargs='+', choices=METRICS, default=list(METRICS))
    p.add_argument('--shared-env', help='Check every scoring component in one existing environment; with --init-config also save this choice')
    p.add_argument('--reuse', action='append', default=[])
    p.add_argument('--conda', default=os.environ.get('CONDA_EXE') or os.environ.get('CONDA', 'conda'))
    p.add_argument('--libraries-only', action='store_true')
    p.add_argument('--check-only', action='store_true', help='Compatibility alias; every check is offline')
    p.add_argument('--no-separation', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--timeout', type=int, default=7200)
    args = p.parse_args(argv)
    try:
        return run_setup(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'[error] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
