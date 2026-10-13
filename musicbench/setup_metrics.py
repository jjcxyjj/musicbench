"""Create isolated metric environments, preload weights, and publish a profile."""
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
            raise ValueError('--reuse must be COMPONENT=ENV, e.g. songeval=my_song_env or clap=/path/to/env')
        result[key] = env.strip()
    return result


def merge_defaults(defaults, overrides):
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_defaults(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def env_args(conda, env):
    return [conda, 'run', '--no-capture-output', '-p' if os.path.isabs(env) else '-n', env]


def base_profile(metrics, envs, repo, runtime, separation=True):
    profile = {'version': 1, 'runtime_environment': runtime, 'metric_environments': {}, 'metric_options': {}}
    for metric in metrics:
        component = 'clap' if metric in ('clap_score', 'fad') else metric
        if metric == 'songeval':
            profile['metric_options'][metric] = {'env': envs[component], 'repo': str(repo), 'device': 'auto'}
        else:
            profile['metric_environments'][metric] = envs[component]
            profile['metric_options'][metric] = {'device': 'auto'}
        if metric in ('clap_score', 'fad'):
            profile['metric_options'][metric]['model'] = 'laion/clap-htsat-fused'
        if metric == 'mulan':
            profile['metric_options'][metric]['model'] = 'OpenMuQ/MuQ-MuLan-large'
        if metric == 'lyrics':
            profile['metric_options'][metric].update(model='large-v3', audio_source='demucs' if separation else 'mix', language='zh')
            if separation:
                profile['metric_options'][metric]['separation'] = {'env': envs['demucs'], 'device': 'auto', 'model': 'htdemucs'}
    return profile


def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding='utf-8')
    temporary.replace(path)


def run_setup(args):
    metrics = list(dict.fromkeys(args.metrics))
    components = components_for(metrics, not args.no_separation)
    reuse = parse_reuse(args.reuse)
    output = Path(args.output).expanduser().resolve()
    existing = yaml.safe_load(output.read_text()) if output.is_file() else {}
    if existing and not isinstance(existing, dict):
        raise ValueError('Existing metric profile must be a YAML mapping')
    existing = existing or {}
    cache = Path(args.cache_dir or ROOT / 'models/metrics').expanduser().resolve()
    runtime = {'HF_HOME': os.environ.get('HF_HOME', str(cache / 'huggingface')),
               'TORCH_HOME': os.environ.get('TORCH_HOME', str(cache / 'torch'))}
    runtime.update(existing.get('runtime_environment', {}))
    if args.cache_dir:
        runtime.update(HF_HOME=str(cache / 'huggingface'), TORCH_HOME=str(cache / 'torch'))
    envs = {}
    for component in components:
        old_env = existing.get('metric_environments', {}).get('clap_score' if component == 'clap' else component)
        if component == 'clap' and not old_env:
            old_env = existing.get('metric_environments', {}).get('fad')
        if component == 'songeval':
            old_env = existing.get('metric_options', {}).get('songeval', {}).get('env')
        if component == 'demucs':
            old_env = existing.get('metric_options', {}).get('lyrics', {}).get('separation', {}).get('env')
        envs[component] = reuse.get(component) or old_env or args.env_prefix + component
    repo = Path(args.songeval_repo or existing.get('metric_options', {}).get('songeval', {}).get('repo') or ROOT / 'third_party/SongEval').expanduser().resolve()
    defaults = base_profile(metrics, envs, repo, runtime, not args.no_separation)
    profile = merge_defaults(defaults, existing)
    profile['runtime_environment'] = runtime
    # Explicit environment selections override old profile entries; other model options survive reruns.
    for metric in metrics:
        component = 'clap' if metric in ('clap_score', 'fad') else metric
        if metric == 'songeval':
            profile.setdefault('metric_options', {}).setdefault(metric, {}).update(env=envs[component], repo=str(repo))
        else:
            profile.setdefault('metric_environments', {})[metric] = envs[component]
    if 'lyrics' in metrics:
        options = profile['metric_options']['lyrics']
        options['audio_source'] = 'mix' if args.no_separation else 'demucs'
        if not args.no_separation:
            options.setdefault('separation', {})['env'] = envs['demucs']
    if args.whisper_model:
        profile['metric_options'].setdefault('lyrics', {})['model'] = args.whisper_model
    print('Selected metrics: ' + ', '.join(metrics))
    print('Shared cache: ' + str(cache))
    mode = 'imports' if args.libraries_only else 'check' if args.check_only else 'load'
    if args.dry_run:
        for component in components:
            print(f'{component}: environment={envs[component]}; template=envs/{COMPONENTS[component]}; mode={mode}')
        if 'songeval' in components:
            print(f'SongEval repo={repo}; clone only if missing (check-only never clones)')
        print(f'Profile after successful setup: {output}')
        return 0
    conda = args.conda
    if not (Path(conda).is_file() or shutil.which(conda)):
        raise ValueError('conda executable not found; pass --conda /absolute/path/to/conda')
    info = subprocess.run([conda, 'env', 'list', '--json'], check=True, capture_output=True, text=True)
    known_paths = {str(Path(p).resolve()) for p in json.loads(info.stdout)['envs']}
    known_names = {Path(p).name for p in known_paths}
    logs = ROOT / 'work/metric-setup' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logs.mkdir(parents=True, exist_ok=False)
    process_env = os.environ.copy()
    process_env.update(profile['runtime_environment'])
    # Installation checks run on CPU and do not compete with generation jobs for CUDA memory.
    process_env['CUDA_VISIBLE_DEVICES'] = ''
    if args.check_only:
        process_env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    status = {'mode': mode, 'components': {}, 'profile': str(output)}

    def run(label, command):
        print(f'[setup] {label}; log={logs / (label + ".log")}', flush=True)
        with (logs / (label + '.log')).open('w') as handle:
            subprocess.run(command, env=process_env, stdout=handle, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)

    for component in components:
        env = envs[component]
        try:
            present = str(Path(env).resolve()) in known_paths if os.path.isabs(env) else env in known_names
            if not present:
                if args.check_only or component in reuse:
                    raise ValueError(f'Expected existing environment is missing: {env}')
                flag = '-p' if os.path.isabs(env) else '-n'
                run('create-' + component, [conda, 'env', 'create', '-f', str(ROOT / 'envs' / COMPONENTS[component]), flag, env])
                if os.path.isabs(env):
                    known_paths.add(str(Path(env).resolve()))
                else:
                    known_names.add(env)
            if component == 'songeval' and mode != 'imports' and not repo.exists():
                if args.check_only:
                    raise ValueError(f'SongEval repo missing: {repo}')
                repo.parent.mkdir(parents=True, exist_ok=True)
                run('clone-songeval', ['git', 'clone', '--depth', '1', 'https://github.com/ASLP-lab/SongEval.git', str(repo)])
            request = {'component': component, 'mode': mode, 'repo': str(repo), 'options': {}}
            if component == 'clap':
                key = 'clap_score' if 'clap_score' in metrics else 'fad'
                request['options'] = profile['metric_options'][key]
            elif component == 'demucs':
                request['options'] = profile['metric_options']['lyrics'].get('separation', {})
            else:
                request['options'] = profile['metric_options'].get(component, {})
            source, destination = logs / f'{component}-request.json', logs / f'{component}-result.json'
            source.write_text(json.dumps(request), encoding='utf-8')
            run('verify-' + component, [*env_args(conda, env), 'python', str(ROOT / 'scripts/prepare_metric.py'),
                                       '--request', str(source), '--output', str(destination)])
            result = json.loads(destination.read_text())
            if result.get('status') != 'ready':
                raise RuntimeError(result.get('reason', 'Readiness check failed'))
            status['components'][component] = {'status': 'ready', 'env': env, **result}
            # Whisper and espeak are saved as explicit local paths for later evaluation.
            if component == 'lyrics':
                if result.get('model_path'):
                    profile['metric_options']['lyrics']['model'] = result['model_path']
                if result.get('espeak_library'):
                    profile['metric_options']['lyrics'].setdefault('environment', {})['PHONEMIZER_ESPEAK_LIBRARY'] = result['espeak_library']
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            status['components'][component] = {'status': 'failed', 'env': env, 'reason': str(exc)}
            print(f'[failed] {component}: {exc}', file=sys.stderr)
        (logs / 'status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False))
    failed = [name for name, entry in status['components'].items() if entry['status'] != 'ready']
    if failed:
        print('Setup incomplete; existing profile preserved. Failed: ' + ', '.join(failed), file=sys.stderr)
        print('Logs: ' + str(logs))
        return 1
    profile['setup_verification'] = {'mode': mode, 'logs': str(logs), 'components': status['components']}
    write_yaml(output, profile)
    print('Metric profile saved: ' + str(output))
    print('Libraries checked; model weights not verified.' if args.libraries_only else 'Models loaded on CPU; validate CUDA on the actual server during evaluation.')
    print('Run ./run.sh; the profile supplies evaluator defaults automatically.')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description='Install metric libraries, download weights, and generate evaluator YAML')
    p.add_argument('--metrics', nargs='+', choices=METRICS, default=list(METRICS))
    p.add_argument('--reuse', action='append', default=[], metavar='COMPONENT=ENV')
    p.add_argument('--conda', default=os.environ.get('CONDA_EXE') or os.environ.get('CONDA', 'conda'))
    p.add_argument('--env-prefix', default='musicbench-eval-')
    p.add_argument('--cache-dir', help='Shared model cache directory; default: project models/metrics')
    p.add_argument('--output', default=str(ROOT / 'configs/metrics.local.yaml'))
    p.add_argument('--songeval-repo')
    p.add_argument('--whisper-model', help='Whisper size/ID or existing local faster-whisper model directory')
    p.add_argument('--no-separation', action='store_true', help='Use mixed audio for lyric ASR; do not install Demucs')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--libraries-only', action='store_true')
    modes.add_argument('--check-only', action='store_true', help='Offline check; do not create environments or clone repositories')
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
