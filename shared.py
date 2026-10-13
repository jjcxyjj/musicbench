"""Local evaluator assets only; never resolve missing weights from remote hubs."""
import copy
import json
import os
from pathlib import Path
import socket


def enforce_offline():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    def blocked(*args, **kwargs):
        raise RuntimeError('Evaluator network access disabled; configure local assets. See README.md')
    # This standalone program only evaluates local audio; model downloads are disabled.
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    socket.gethostbyname = blocked
    socket.gethostbyname_ex = blocked


def local_file(value, label):
    path = Path(value or '/__musicbench_unconfigured__').expanduser()
    if not value or not path.is_file():
        raise ValueError(f'{label}: local file missing: {value!r}; see README.md')
    return str(path.resolve())


def local_directory(value, label, config=True, weights=True):
    path = Path(value or '/__musicbench_unconfigured__').expanduser()
    if not value or not path.is_dir():
        raise ValueError(f'{label}: local directory missing: {value!r}; model IDs are not accepted. See README.md')
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


def device_name(requested='auto'):
    import torch
    return ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested


def summarize(rows, keys, expected, errors):
    import math
    scores = {}
    for key in keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
        scores[key] = sum(values) / len(values) if values else None
    return {'status': 'completed' if len(rows) == expected and not errors else 'partial' if rows else 'failed',
            'scores': scores, 'per_sample': rows, 'errors': errors,
            'n_expected': expected, 'n_evaluated': len(rows)}
