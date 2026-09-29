"""dsp_beat inference script.

Runs in the algorithm's own conda env. It does NOT import musicbench — the
interface is: read manifest.json, write predictions.jsonl.

    python infer.py --manifest dataset/manifest.json --output preds/dsp_beat.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np


def read_wav(path: str):
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def rms_envelope(audio: np.ndarray, sr: int, frame: int = 1024, hop: int = 256) -> np.ndarray:
    n = len(audio)
    env: List[float] = []
    for start in range(0, n - frame + 1, hop):
        seg = audio[start:start + frame]
        env.append(float(np.sqrt(np.mean(seg ** 2)) + 1e-12))
    return np.asarray(env, dtype=float)


def peak_pick(env: np.ndarray, sr: int, min_interval: float = 0.3, threshold_std: float = 0.5) -> List[float]:
    hop = 256
    min_gap = max(1, int(min_interval * sr / hop))
    thresh = np.mean(env) + threshold_std * np.std(env)
    beats: List[float] = []
    i = 0
    while i < len(env):
        if env[i] >= thresh:
            window = env[i:i + min_gap]
            j = int(np.argmax(window))
            peak_idx = i + j
            beats.append(peak_idx * hop / sr)
            i = peak_idx + min_gap
        else:
            i += 1
    return beats


def tempo_from_beats(beats: List[float]) -> float:
    if len(beats) < 2:
        return float("nan")
    ibis = np.diff(np.asarray(beats))
    ibis = ibis[(ibis >= 0.2) & (ibis <= 2.0)]
    if ibis.size == 0:
        return float("nan")
    return float(60.0 / np.median(ibis))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    base = os.path.dirname(args.manifest)
    lines: List[str] = []
    for row in manifest:
        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(base, audio_path)
        audio, sr = read_wav(audio_path)
        env = rms_envelope(audio, sr)
        beats = peak_pick(env, sr)
        tempo = tempo_from_beats(beats)
        out: Dict[str, Any] = {"id": row["id"], "output": {"beats": beats, "tempo": tempo}}
        lines.append(json.dumps(out, ensure_ascii=False))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[dsp_beat] wrote {len(lines)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
