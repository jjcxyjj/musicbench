"""SongEval aesthetics metric (no-reference, five perceptual dimensions).

Predicts professional musician-aligned aesthetic scores for generated audio
using the SongEval toolkit (https://github.com/ASLP-lab/SongEval): a small
attention regressor over MuQ hidden states, following arXiv:2505.10793.

The five dimensions (all higher-is-better, 1..5 scale):

    coherence    overall coherence
    musicality   overall musicality
    memorability memorability
    clarity      clarity of song structure
    naturalness  naturalness of vocal breathing & phrasing

Dependencies (all optional — the metric degrades gracefully):

    pip install muq safetensors librosa torch

Set ``MUSICBENCH_SONGEVAL_CKPT`` to the path of the SongEval
``ckpt/model.safetensors`` file (default ``ckpt/model.safetensors``), and
``MUSICBENCH_SONGEVAL_MUQ`` to override the MuQ checkpoint
(default ``OpenMuQ/MuQ-large-msd-iter``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.data import Prediction, Sample
from ..core.metric import Metric
from ..core.registry import METRICS

# Score order produced by the SongEval Generator (see its eval.py).
DIMENSIONS = ["coherence", "musicality", "memorability", "clarity", "naturalness"]


def _generated_paths(predictions: List[Prediction]) -> List[str]:
    out = []
    for p in predictions:
        if isinstance(p.output, dict):
            ap = p.output.get("audio_path")
            if ap and os.path.isfile(ap):
                out.append(ap)
    return out


def _build_generator() -> Any:
    """Build the SongEval regressor (a small attention model). torch is imported
    lazily so importing this module never requires torch at import time."""
    import torch
    import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self, in_features=1024, ffd_hidden_size=4096,
                     num_classes=5, attn_layer_num=4):
            super().__init__()
            self.attn = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=in_features, num_heads=8,
                                      dropout=0.2, batch_first=True)
                for _ in range(attn_layer_num)
            ])
            self.ffd = nn.Sequential(
                nn.Linear(in_features, ffd_hidden_size),
                nn.ReLU(),
                nn.Linear(ffd_hidden_size, in_features),
            )
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(in_features * 2, num_classes)
            self.proj = nn.Tanh()

        def forward(self, ssl_feature, judge_id=None):
            # ssl_feature: [B, T, D] -> [B, num_classes]
            ssl_feature = self.ffd(ssl_feature)
            tmp = ssl_feature
            for attn in self.attn:
                tmp, _ = attn(tmp, tmp, tmp)
            ssl_feature = self.dropout(torch.concat([
                torch.mean(tmp, dim=1),
                torch.max(ssl_feature, dim=1)[0],
            ], dim=1))
            x = self.fc(ssl_feature)
            return self.proj(x) * 2.0 + 3

    return Generator()


def _load_songeval() -> Optional[Dict[str, Any]]:
    """Load the SongEval regressor + MuQ encoder, or return None on any failure."""
    try:
        import torch
        from muq import MuQ
        from safetensors.torch import load_file
    except Exception:
        return None

    ckpt = os.environ.get("MUSICBENCH_SONGEVAL_CKPT", "ckpt/model.safetensors")
    if not os.path.isfile(ckpt):
        return None

    try:
        generator = _build_generator()
        generator.load_state_dict(load_file(ckpt, device="cpu"), strict=False)
        generator.eval()

        muq_id = os.environ.get("MUSICBENCH_SONGEVAL_MUQ", "OpenMuQ/MuQ-large-msd-iter")
        muq = MuQ.from_pretrained(muq_id)
        muq.eval()

        device = os.environ.get("MUSICBENCH_SONGEVAL_DEVICE")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        return {
            "generator": generator.to(device),
            "muq": muq.to(device),
            "device": device,
        }
    except Exception:
        return None


def _predict(bundle: Dict[str, Any], path: str) -> List[float]:
    import librosa
    import torch

    device = bundle["device"]
    wav, sr = librosa.load(path, sr=24000)
    audio = torch.tensor(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        out = bundle["muq"](audio, output_hidden_states=True)
        feature = out["hidden_states"][6]
        scores = bundle["generator"](feature).squeeze(0)
    return [round(float(v), 4) for v in scores.cpu().tolist()]


@METRICS.register("songeval")
class SongEvalMetric(Metric):
    """Five-dimensional aesthetic scores of generated audio (no-reference).

    Higher is better for every dimension. Returns ``{"available": false,
    "reason": ...}`` when MuQ/SongEval weights are unavailable so the rest of
    the benchmark can still complete.
    """

    name = "songeval"

    def compute(self, predictions: List[Prediction], samples: List[Sample]) -> Dict[str, Any]:
        paths = _generated_paths(predictions)
        if not paths:
            return {"available": False, **{d: float("nan") for d in DIMENSIONS},
                    "reason": "no generated audio"}

        bundle = _load_songeval()
        if bundle is None:
            return {"available": False, **{d: float("nan") for d in DIMENSIONS},
                    "reason": "SongEval/MuQ unavailable (pip install muq safetensors "
                              "and set MUSICBENCH_SONGEVAL_CKPT)"}

        acc: Dict[str, List[float]] = {d: [] for d in DIMENSIONS}
        n = 0
        for p in predictions:
            ap = p.output.get("audio_path") if isinstance(p.output, dict) else None
            if not ap or not os.path.isfile(ap):
                continue
            try:
                vals = _predict(bundle, ap)
            except Exception:
                continue
            for d, v in zip(DIMENSIONS, vals):
                acc[d].append(v)
            n += 1

        if n == 0:
            return {"available": False, **{d: float("nan") for d in DIMENSIONS},
                    "reason": "no valid audio scored"}

        out: Dict[str, Any] = {"available": True, "n_evaluated": n}
        for d in DIMENSIONS:
            out[d] = float(np.mean(acc[d]))
        out["mean"] = float(np.mean([out[d] for d in DIMENSIONS]))
        return out