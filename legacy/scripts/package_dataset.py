"""Package a dataset (audio + manifest.json) for algorithm inference.

The packaged layout is:

    output_dir/
    ├── manifest.json          # [{id, audio_path, ground_truth, ...}, ...]
    └── audio/                 # the wav files

Algorithms read manifest.json and write predictions.jsonl; they do NOT need to
import musicbench (the JSON contract is the only interface).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, List

import musicbench.datasets  # noqa: F401  (register datasets)

from musicbench.core.data import _jsonable
from musicbench.core.registry import get_dataset


def package(dataset_name: str, split: str, output_dir: str, **dataset_kwargs: Any) -> List[Dict[str, Any]]:
    cls = get_dataset(dataset_name)
    ds = cls(**dataset_kwargs) if dataset_kwargs else cls()
    samples = ds.load(split)

    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    for index, s in enumerate(samples):
        if not s.audio_path or not os.path.isfile(s.audio_path):
            # Keep samples whose reference audio is missing so the manifest
            # still lists them; algorithms/metrics may skip them.
            manifest.append(
                {
                    "id": s.id,
                    "audio_path": s.audio_path or "",
                    "metadata": _jsonable(s.metadata),
                    "ground_truth": _jsonable(s.ground_truth),
                }
            )
            continue
        ext = os.path.splitext(s.audio_path)[1] or ".wav"
        dst = os.path.join(audio_dir, f"sample_{index:06d}{ext}")
        shutil.copy2(s.audio_path, dst)
        manifest.append(
            {
                "id": s.id,
                "audio_path": os.path.relpath(dst, output_dir),
                "metadata": _jsonable(s.metadata),
                "ground_truth": _jsonable(s.ground_truth),
            }
        )

    if hasattr(ds, "selection_summary"):
        with open(os.path.join(output_dir, "sampling.json"), "w", encoding="utf-8") as f:
            json.dump(ds.selection_summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Package a dataset for algorithm inference")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True)
    # synthetic dataset params
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--tempo", type=float, default=120.0)
    # feature_json dataset params
    p.add_argument("--root", default="/data/MusicLite_data/feature")
    p.add_argument("--json-name", default="5_with_latent.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling-strategy", choices=["date_balanced", "pooled"], default="date_balanced")
    p.add_argument("--audio-field", default="path")
    p.add_argument("--caption-field", default="Qwen_caption")
    p.add_argument("--lyrics-field", default="lyrics")
    p.add_argument("--language-field", default="language")
    args = p.parse_args()

    if args.dataset == "feature_json":
        manifest = package(
            args.dataset, args.split, args.output,
            root=args.root, num_samples=args.num_samples, json_name=args.json_name,
            seed=args.seed, sampling_strategy=args.sampling_strategy, audio_field=args.audio_field,
            caption_field=args.caption_field, lyrics_field=args.lyrics_field, language_field=args.language_field,
        )
    else:
        manifest = package(
            args.dataset, args.split, args.output,
            num_samples=args.num_samples, duration=args.duration, sr=args.sr, tempo=args.tempo,
        )
    print(f"[package] wrote {len(manifest)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
