"""Run one evaluator readiness check inside its own environment (CPU, serial)."""
import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULES = {
    'audiobox': ['torch', 'audiobox_aesthetics.infer'],
    'clap': ['torch', 'transformers', 'librosa'],
    'mulan': ['torch', 'muq', 'librosa'],
    'lyrics': ['torch', 'faster_whisper', 'pypinyin', 'phonemizer'],
    'demucs': ['torch', 'demucs.pretrained'],
    'songeval': ['torch', 'librosa', 'muq', 'omegaconf', 'safetensors.torch', 'einops'],
}


def espeak_path():
    if os.environ.get('PHONEMIZER_ESPEAK_LIBRARY'):
        return os.environ['PHONEMIZER_ESPEAK_LIBRARY']
    library = Path(sys.prefix) / 'lib'
    matches = sorted(library.glob('libespeak-ng.so*')) + sorted(library.glob('libespeak-ng*.dylib'))
    if matches:
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(matches[0].resolve())
        return os.environ['PHONEMIZER_ESPEAK_LIBRARY']
    return None


def verify(job):
    component, mode = job['component'], job['mode']
    options = job.get('options', {})
    os.environ.update(options.get('environment', {}))
    from musicbench.evaluation.offline import enforce_offline, validate_assets, local_directory, load_mulan, load_muq
    enforce_offline()
    errors = {}
    for module in MODULES[component]:
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors[module] = f'{type(exc).__name__}: {exc}'
    if errors:
        return {'status': 'failed', 'component': component, 'missing_dependencies': list(errors),
                'dependency_errors': errors, 'reason': 'Missing or broken libraries; see docs/METRIC_SETUP.md'}
    import torch
    result = {'status': 'ready', 'component': component, 'mode': mode,
              'python': sys.executable, 'torch_version': torch.__version__, 'load_device': 'cpu',
              'packages': {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}}
    if component == 'lyrics':
        library = espeak_path()
        from phonemizer import phonemize
        from musicbench.evaluation.lyrics import phonemes
        if not phonemize('hello world', language='en-us', backend='espeak').strip():
            raise RuntimeError('English espeak phonemizer produced no phones')
        if not phonemes('你好', 'zh'):
            raise RuntimeError('Chinese G2P produced no phones')
        result['espeak_library'] = library
    if mode == 'imports':
        result['weights_verified'] = False
        return result
    validate_assets(component, options)
    if component == 'audiobox':
        from audiobox_aesthetics.infer import initialize_predictor
        predictor = initialize_predictor(ckpt=options['checkpoint'])
        predictor.device = torch.device('cpu')
        predictor.model = predictor.model.to(predictor.device).eval()
        result['model'] = options.get('checkpoint') or 'facebook/audiobox-aesthetics'
        del predictor
    elif component == 'clap':
        from transformers import ClapModel, ClapProcessor
        name = options['model']
        model = ClapModel.from_pretrained(name, local_files_only=True).to('cpu').eval()
        processor = ClapProcessor.from_pretrained(name, local_files_only=True)
        with torch.inference_mode():
            features = model.get_text_features(**processor(text=['piano solo'], return_tensors='pt'))
        if not torch.isfinite(features).all():
            raise RuntimeError('CLAP returned non-finite text features')
        result['model'] = name
    elif component == 'mulan':
        from muq import MuQMuLan
        name = options['model']
        model = load_mulan(options).to('cpu').eval()
        # Text encoding exercises the dependent text encoder and tokenizer caches too.
        with torch.inference_mode():
            features = model(texts=['piano solo'])
        if not torch.isfinite(features).all():
            raise RuntimeError('MuQ-MuLan returned non-finite text features')
        result['model'] = name
    elif component == 'lyrics':
        from faster_whisper import WhisperModel
        name = options['model']
        path = name
        model = WhisperModel(path, device='cpu', compute_type='int8', local_files_only=True)
        result.update(model=name, model_path=str(Path(path).resolve()))
    elif component == 'demucs':
        from demucs.pretrained import get_model
        name = options.get('model', 'htdemucs')
        model = get_model(name, repo=Path(options['repo'])).to('cpu').eval()
        result['model'] = name
    elif component == 'songeval':
        from muq import MuQ
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        repo = Path(options['repo']).resolve()
        checkpoint = repo / 'ckpt/model.safetensors'
        if not (repo / 'model.py').is_file() or not checkpoint.is_file():
            raise ValueError(f'SongEval code or checkpoint missing: {repo}')
        with checkpoint.open('rb') as handle:
            if handle.read(128).startswith(b'version https://git-lfs.github.com/spec'):
                raise ValueError('SongEval checkpoint is a Git LFS pointer. Install git-lfs and run git lfs pull in that repo')
        spec = importlib.util.spec_from_file_location('musicbench_setup_songeval', repo / 'model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = OmegaConf.to_container(OmegaConf.load(repo / 'config.yaml'))['generator']
        config.pop('_target_', None)
        model = module.Generator(**config)
        model.load_state_dict(load_file(str(checkpoint), device='cpu'), strict=True)
        encoder = load_muq(options.get('muq_model')).to('cpu').eval()
        digest = hashlib.sha256()
        with checkpoint.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        result.update(repo=str(repo), checkpoint_sha256=digest.hexdigest(), muq=options.get('muq_model'))
    result['weights_verified'] = True
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    job = json.loads(Path(args.request).read_text())
    try:
        result = verify(job)
    except Exception as exc:
        traceback.print_exc()
        result = {'status': 'failed', 'component': job['component'], 'reason': f'{type(exc).__name__}: {exc}'}
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
