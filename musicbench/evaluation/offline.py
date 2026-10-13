"""Local evaluator assets only; never resolve missing weights from remote hubs."""
import copy
import json
import os
from pathlib import Path
import socket


def enforce_offline():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    def blocked(*args, **kwargs):
        raise RuntimeError('Evaluator network access disabled; configure local assets. See docs/METRIC_SETUP.md')
    # Only called in dedicated evaluator workers, never in the algorithm process.
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    socket.gethostbyname = blocked
    socket.gethostbyname_ex = blocked


def local_file(value, label):
    path = Path(value or '/__musicbench_unconfigured__').expanduser()
    if not value or not path.is_file():
        raise ValueError(f'{label}: local file missing: {value!r}; see docs/METRIC_SETUP.md')
    return str(path.resolve())


def local_directory(value, label, config=True, weights=True):
    path = Path(value or '/__musicbench_unconfigured__').expanduser()
    if not value or not path.is_dir():
        raise ValueError(f'{label}: local directory missing: {value!r}; model IDs are not accepted. See docs/METRIC_SETUP.md')
    if config:
        local_file(path / 'config.json', label + '.config.json')
    if weights:
        if not any(list(path.glob(pattern)) for pattern in ('*.safetensors', '*model*.bin', '*.pt')):
            raise ValueError(f'{label}: local model weights missing in {path}')
    return str(path.resolve())


def mulan_config(options):
    model = local_directory(options.get('model'), 'mulan.model')
    data = json.loads((Path(model) / 'config.json').read_text())
    config = copy.deepcopy(data.get('config', data))
    for key in ('audio_model', 'text_model'):
        value = options.get(key) or config.get(key, {}).get('name')
        if value and not Path(value).is_absolute() and (Path(model) / value).is_dir():
            value = str(Path(model) / value)
        config[key]['name'] = local_directory(value, 'mulan.' + key)
    # Upstream selects model classes by these names, including for local paths.
    if 'muq' not in config['audio_model']['name'].lower():
        raise ValueError('MuQ-MuLan audio_model directory name must contain muq (upstream model dispatch)')
    if 'xlm-roberta' not in config['text_model']['name']:
        raise ValueError('Official MuQ-MuLan text_model directory name must contain xlm-roberta')
    return model, config


def load_mulan(options):
    model, config = mulan_config(options)
    from muq import MuQMuLan
    return MuQMuLan.from_pretrained(model, config=config, local_files_only=True)


def load_muq(value):
    path = local_directory(value, 'songeval.muq_model')
    from muq import MuQ
    return MuQ.from_pretrained(path, local_files_only=True)


def validate_assets(component, options):
    if component == 'audiobox':
        local_file(options.get('checkpoint'), 'audiobox.checkpoint')
    elif component == 'clap':
        local_directory(options.get('model'), 'clap.model')
    elif component == 'mulan':
        mulan_config(options)
    elif component == 'lyrics':
        path = local_directory(options.get('model'), 'lyrics.model', weights=False)
        local_file(Path(path) / 'model.bin', 'lyrics.model.bin')
        local_file(Path(path) / 'tokenizer.json', 'lyrics.tokenizer.json')
    elif component == 'demucs':
        path = local_directory(options.get('repo'), 'lyrics.separation.repo', config=False, weights=False)
        name = options.get('model', 'htdemucs')
        yaml_path = Path(path) / (name + '.yaml')
        if not yaml_path.is_file() and not list(Path(path).glob(name + '-*.th')):
            raise ValueError(f'Demucs local repo missing {name}.yaml or {name}-*.th: {path}')
    elif component == 'songeval':
        repo = local_directory(options.get('repo'), 'songeval.repo', config=False, weights=False)
        for name in ('model.py', 'config.yaml', 'ckpt/model.safetensors'):
            local_file(Path(repo) / name, 'songeval.' + name)
        local_directory(options.get('muq_model'), 'songeval.muq_model')
