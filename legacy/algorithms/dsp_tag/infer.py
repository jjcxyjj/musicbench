"""dsp_tag inference script.

Runs in the algorithm's own conda env. Reads manifest.json, writes
predictions.jsonl (soft tag scores).

    python infer.py --manifest dataset/manifest.json --output preds/dsp_tag.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np


def spectral_centroid(seg: np.ndarray, sr: int) -> float:
    n = len(seg)
    if n == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    denom = spec.sum()
    if denom < 1e-12:
        return 0.0
    return float((freqs * spec).sum() / denom)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from scipy.io import wavfile

    with open(args.manifest) as f:
        manifest = json.load(f)

    base = os.path.dirname(args.manifest)
    lines: List[str] = []
    for row in manifest:
        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(base, audio_path)
        sr, data = wavfile.read(audio_path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        else:
            data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        seg = data[: min(len(data), sr)]
        centroid = spectral_centroid(seg, sr)
        d_low, d_high = abs(centroid - 150.0), abs(centroid - 2000.0)
        total = d_low + d_high + 1e-9
        out: Dict[str, Any] = {
            "id": row["id"],
            "output": {"scores": {"low": float(d_high / total), "high": float(d_low / total)}},
        }
        lines.append(json.dumps(out, ensure_ascii=False))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[dsp_tag] wrote {len(lines)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
