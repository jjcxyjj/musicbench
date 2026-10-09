"""Dataset loader for the MusicLite feature JSON layout.

Scans ``<root>/*/5_with_latent.json`` (one JSON file per time-stamped batch),
concatenates the records, and randomly samples ``num_samples`` of them.

Each record looks like::

    {
      "path": "/data/.../audio/xxx.wav",          # audio to evaluate
      "path_original": "/data/.../xxx.wav",        # original audio
      "has_vocal": true,
      "vocal_score": 0.21,
      "vocal_keywords": [...],
      "Qwen_caption": "...",                        # text prompt
      "gf_style_name": "国风流行",
      "gf_style_code": "gflx",
      "dynamics_path": "...", "melody_path": "...", "rhythm_path": "...",
      "latent_path": "...",
    }

The ``Sample`` carries the audio path as ``audio_path`` (reference audio) and
the caption/style/vocal info as ``metadata``. For text2music evaluation the
caption is the prompt and the reference audio is the target.
"""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Any, Dict, List, Optional

from ..core.data import Sample
from ..core.dataset import Dataset
from ..core.registry import DATASETS


@DATASETS.register("feature_json")
class FeatureJsonDataset(Dataset):
    name = "feature_json"

    def __init__(
        self,
        root: str = "/data/MusicLite_data/feature",
        num_samples: int = 10,
        seed: int = 0,
        json_name: str = "5_with_latent.json",
        audio_field: str = "path",
        caption_field: str = "Qwen_caption",
    ) -> None:
        self.root = root
        self.num_samples = int(num_samples)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.seed = int(seed)
        self.json_name = json_name
        self.audio_field = audio_field
        self.caption_field = caption_field

    # ------------------------------------------------------------------ load
    def load(self, split: str = "test") -> List[Sample]:
        records = self._collect_records()
        rng = random.Random(self.seed)
        if len(records) > self.num_samples:
            records = rng.sample(records, self.num_samples)
        samples = [self._to_sample(i, rec) for i, rec in enumerate(records)]
        return samples

    # ------------------------------------------------------------- internals
    def _collect_records(self) -> List[Dict[str, Any]]:
        pattern = os.path.join(self.root, "*", self.json_name)
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No feature JSON files matching {pattern}")
        records: List[Dict[str, Any]] = []
        for fp in files:
            try:
                with open(fp) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read feature JSON {fp}: {exc}") from exc
            # Support both a JSON array and a single object.
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
            else:
                raise ValueError(f"Expected JSON array or object: {fp}")
        if not records:
            raise ValueError("Feature dataset contains no samples")
        return records

    def _to_sample(self, idx: int, rec: Dict[str, Any]) -> Sample:
        audio_path = rec.get(self.audio_field) or rec.get("path") or ""
        caption = rec.get(self.caption_field) or rec.get("Qwen_caption") or ""
        # Keep the rest of the record as metadata for L1 metrics / future use.
        metadata = {
            "caption": caption,
            "has_vocal": rec.get("has_vocal"),
            "vocal_score": rec.get("vocal_score"),
            "style_name": rec.get("gf_style_name"),
            "style_code": rec.get("gf_style_code"),
            "path_original": rec.get("path_original"),
            "dynamics_path": rec.get("dynamics_path"),
            "melody_path": rec.get("melody_path"),
            "rhythm_path": rec.get("rhythm_path"),
            "latent_path": rec.get("latent_path"),
        }
        sample_id = rec.get("id") or f"ml_{os.path.basename(audio_path)}_{idx}"
        return Sample(
            id=str(sample_id),
            audio_path=audio_path,
            metadata={k: v for k, v in metadata.items() if v is not None},
            ground_truth={"caption": caption, "has_vocal": rec.get("has_vocal")},
        )
