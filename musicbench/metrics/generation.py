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
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _load_clap() -> Optional[Any]:
    """Load a CLAP model (audio embedding) via transformers, if available."""
    try:
        from transformers import ClapModel, ClapProcessor  # type: ignore

        ckpt = os.environ.get("MUSICBENCH_CLAP_MODEL", "laion/clap-htsat-fused")
        model = ClapModel.from_pretrained(ckpt).eval()
        model.requires_grad_(False)
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


def _fad(emb_ref: np.ndarray, emb_gen: np.ndarray) -> float:
    mu_r = emb_ref.mean(axis=0)
    mu_g = emb_gen.mean(axis=0)
    cov_r = np.cov(emb_ref, rowvar=False)
    cov_g = np.cov(emb_gen, rowvar=False)
    diff = mu_r - mu_g
    # Fréchet distance = ||mu_r-mu_g||^2 + Tr(C_r + C_g - 2*(C_r C_g)^{1/2})
    root_r = _sqrtm(cov_r)
    covmean = _sqrtm(root_r @ cov_g @ root_r)
    tr = np.trace(cov_r + cov_g - 2 * covmean)
    return max(0.0, float(diff @ diff + tr))


def _sqrtm(a: np.ndarray) -> np.ndarray:
    """Matrix square root (real) via eigen-decomposition."""
    w, v = np.linalg.eigh((a + a.T) / 2)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.T


@METRICS.register("fad")
class FADMetric(Metric):
    """Fréchet Audio Distance between generated and reference embeddings.

    Lower is better. Requires a CLAP model (``transformers`` + ``librosa``).
    Set ``MUSICBENCH_CLAP_MODEL`` to override the checkpoint.
    """

    name = "fad"

    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        gen = _generated_paths(predictions)
        ref = _reference_paths(samples)
        if len(gen) < 2 or len(ref) < 2:
            return {"available": False, "fad": float("nan"), "reason": "insufficient audio"}

        clap = _load_clap()
        if clap is None:
            return {"available": False, "fad": float("nan"),
                    "reason": "CLAP model unavailable (install transformers+librosa)"}

        try:
            emb_ref = _embed_clap(clap, ref)
            emb_gen = _embed_clap(clap, gen)
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
            p_ref = np.mean([at.inference(_load(p)[None, :])[0][0] for p in ref], axis=0)
            p_gen = np.mean([at.inference(_load(p)[None, :])[0][0] for p in gen], axis=0)
            eps = 1e-9
            p_ref = np.maximum(p_ref, eps)
            p_gen = np.maximum(p_gen, eps)
            p_ref = p_ref / p_ref.sum()
            p_gen = p_gen / p_gen.sum()
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
        return {"available": True, "clap": float(np.mean(sims)),
                "n_evaluated": len(sims),
                "coverage": len(sims) / len(samples) if samples else 0.0}


def _load(path: str) -> np.ndarray:
    from ..core.audio_features import load_mono

    x, _ = load_mono(path, 32000)
    return x
