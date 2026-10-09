"""L1 "professional" metrics for text-to-music (deep models, optional).

These require embedding/classification models (VGGish / PANNs / CLAP). They
degrade gracefully: if the dependency or model is unavailable, they return
``{"available": false, "reason": ...}`` instead of crashing, so a "basic"
benchmark can still complete without them.

Model download uses ``HF_ENDPOINT=https://hf-mirror.com`` when set (China
mirror); see each function's docstring.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.data import Prediction, Sample
from ..core.metric import Metric
from ..core.registry import METRICS


def _generated_paths(predictions: List[Prediction]) -> List[str]:
    out = []
    for p in predictions:
        if isinstance(p.output, dict):
            ap = p.output.get("audio_path")
            if ap and os.path.isfile(ap):
                out.append(ap)
    return out


def _reference_paths(samples: List[Sample]) -> List[str]:
    return [s.audio_path for s in samples if os.path.isfile(s.audio_path)]


def _load_clap() -> Optional[Any]:
    """Load a CLAP model (audio embedding) via transformers, if available."""
    try:
        from transformers import ClapModel, ClapProcessor  # type: ignore

        ckpt = os.environ.get("MUSICBENCH_CLAP_MODEL", "laion/clap-htsat-fused")
        model = ClapModel.from_pretrained(ckpt)
        processor = ClapProcessor.from_pretrained(ckpt)
        return {"model": model, "processor": processor, "ckpt": ckpt}
    except Exception as exc:  # pragma: no cover - dependency optional
        return None


def _embed_clap(clap: Dict[str, Any], audio_paths: List[str]) -> np.ndarray:
    from transformers import ClapModel, ClapProcessor  # type: ignore

    model: ClapModel = clap["model"]
    processor: ClapProcessor = clap["processor"]
    import librosa

    feats = []
    for p in audio_paths:
        audio, sr = librosa.load(p, sr=48000, mono=True)
        inputs = processor(audios=audio, sampling_rate=48000, return_tensors="pt")
        emb = model.get_audio_features(**inputs)
        feats.append(emb.detach().cpu().numpy().squeeze(0))
    return np.stack(feats)


def _load_vggish() -> Optional[Any]:
    """Load a VGGish embedding model via ``torchvggish``, if available.

    VGGish (128-d Audioset embeddings) is the canonical embedding for
    Fréchet Audio Distance. ``torchvggish`` downloads its weights from
    GitHub releases on first use; set the proxy or pre-download if needed.
    """
    try:
        import torchvggish  # type: ignore

        model = torchvggish.vggish()
        model.eval()
        return model
    except Exception:
        return None


def _embed_vggish(model: Any, audio_paths: List[str]) -> np.ndarray:
    """Embed each audio file with VGGish and mean-pool its frame embeddings."""
    from torchvggish import vggish_input  # type: ignore

    import librosa

    feats = []
    for p in audio_paths:
        audio, sr = librosa.load(p, sr=None, mono=True)
        examples = vggish_input.waveform_to_examples(audio, sr)
        emb = model(examples).detach().cpu().numpy()
        emb = np.atleast_2d(emb)
        feats.append(emb.mean(axis=0))
    return np.stack(feats)


def _fad(emb_ref: np.ndarray, emb_gen: np.ndarray) -> float:
    mu_r = emb_ref.mean(axis=0)
    mu_g = emb_gen.mean(axis=0)
    cov_r = np.cov(emb_ref, rowvar=False)
    cov_g = np.cov(emb_gen, rowvar=False)
    diff = mu_r - mu_g
    # Fréchet distance = ||mu_r-mu_g||^2 + Tr(C_r + C_g - 2*(C_r C_g)^{1/2})
    covmean = _sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr = np.trace(cov_r + cov_g - 2 * covmean)
    return float(diff @ diff + tr)


def _sqrtm(a: np.ndarray) -> np.ndarray:
    """Matrix square root (real) via eigen-decomposition."""
    w, v = np.linalg.eigh(a)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.T


@METRICS.register("fad")
class FADMetric(Metric):
    """Fréchet Audio Distance between generated and reference VGGish embeddings.

    Lower is better. Uses standard VGGish embeddings (``torchvggish``), matching
    the original FAD definition, instead of CLAP embeddings. Degrades gracefully
    (``{"available": false}``) when torch/torchvggish is unavailable.
    """

    name = "fad"

    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        gen = _generated_paths(predictions)
        ref = _reference_paths(samples)
        if not gen or len(ref) < 2:
            return {"available": False, "fad": float("nan"), "reason": "insufficient audio"}

        model = _load_vggish()
        if model is None:
            return {"available": False, "fad": float("nan"),
                    "reason": "VGGish unavailable (pip install torchvggish)"}

        try:
            emb_ref = _embed_vggish(model, ref)
            emb_gen = _embed_vggish(model, gen)
            return {"available": True, "fad": _fad(emb_ref, emb_gen)}
        except Exception as exc:
            return {"available": False, "fad": float("nan"), "reason": str(exc)[:200]}


@METRICS.register("kl_div")
class KLDivergenceMetric(Metric):
    """KL divergence of tag distributions (reference vs generated).

    Lower is better. Requires PANNs (``panns_inference``). Because PANNs
    produces a tag posterior per clip, we compare the averaged posteriors.
    """

    name = "kl_div"

    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        gen = _generated_paths(predictions)
        ref = _reference_paths(samples)
        if not gen or not ref:
            return {"available": False, "kl": float("nan"), "reason": "insufficient audio"}
        try:
            from panns_inference import AudioTagging  # type: ignore
        except Exception:
            return {"available": False, "kl": float("nan"),
                    "reason": "panns_inference unavailable (pip install panns-inference)"}

        try:
            at = AudioTagging(checkpoint_path=None, device="cpu")
            p_ref = np.mean(at.inference(np.concatenate([_load(p) for p in ref]))[0], axis=0)
            p_gen = np.mean(at.inference(np.concatenate([_load(p) for p in gen]))[0], axis=0)
            eps = 1e-9
            kl = float(np.sum(p_ref * np.log((p_ref + eps) / (p_gen + eps))))
            return {"available": True, "kl": kl}
        except Exception as exc:
            return {"available": False, "kl": float("nan"), "reason": str(exc)[:200]}


@METRICS.register("clap_score")
class CLAPScoreMetric(Metric):
    """CLAP text-audio alignment: cosine sim between caption and generated audio.

    Higher is better. Uses the sample caption (metadata) vs generated audio.
    """

    name = "clap_score"

    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        clap = _load_clap()
        if clap is None:
            return {"available": False, "clap": float("nan"),
                    "reason": "CLAP model unavailable"}

        from transformers import ClapModel, ClapProcessor  # type: ignore
        import librosa

        model: ClapModel = clap["model"]
        processor: ClapProcessor = clap["processor"]

        by_id = {s.id: s for s in samples}
        sims = []
        for p in predictions:
            s = by_id.get(p.id)
            if s is None:
                continue
            caption = (s.metadata or {}).get("caption") or (s.metadata or {}).get("Qwen_caption")
            gen_path = p.output.get("audio_path") if isinstance(p.output, dict) else None
            if not caption or not gen_path or not os.path.isfile(gen_path):
                continue
            try:
                audio, sr = librosa.load(gen_path, sr=48000, mono=True)
                a_in = processor(audios=audio, sampling_rate=48000, return_tensors="pt")
                t_in = processor(text=[caption], return_tensors="pt")
                a_emb = model.get_audio_features(**a_in)
                t_emb = model.get_text_features(**t_in)
                a_emb = a_emb / a_emb.norm(dim=-1, keepdim=True)
                t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
                sims.append(float((a_emb * t_emb).sum().item()))
            except Exception:
                continue
        if not sims:
            return {"available": False, "clap": float("nan"), "reason": "no valid caption+audio pairs"}
        return {"available": True, "clap": float(np.mean(sims))}


def _load(path: str) -> np.ndarray:
    from ..core.audio_features import load_mono

    x, _ = load_mono(path, 32000)
    return x
