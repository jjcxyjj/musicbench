"""Optional SongEval integration through an isolated Python worker."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ..core.metric import Metric
from ..core.registry import METRICS

DIMENSIONS = ("coherence", "musicality", "memorability", "clarity", "naturalness")


@METRICS.register("songeval")
class SongEvalMetric(Metric):
    name = "songeval"

    def compute(self, predictions, samples):
        root = Path(__file__).resolve().parents[2]
        repo = Path(os.environ.get("MUSICBENCH_SONGEVAL_REPO", root / "third_party/SongEval")).resolve()
        if not (repo / "ckpt/model.safetensors").is_file():
            return {"available": False, "reason": "SongEval checkpoint missing; see docs/SONGEVAL.md"}
        by_id = {s.id: s for s in samples}
        rows = []
        for p in predictions:
            if p.id in by_id:
                rows.append({"id": p.id, "audio_path": p.output.get("audio_path", ""),
                             "has_vocal": by_id[p.id].metadata.get("has_vocal")})
        if not rows:
            return {"available": False, "reason": "No matched predictions"}
        with tempfile.TemporaryDirectory(prefix="musicbench-songeval-") as tmp:
            src, dst = Path(tmp) / "input.json", Path(tmp) / "output.json"
            src.write_text(json.dumps(rows), encoding="utf-8")
            command = [os.environ.get("MUSICBENCH_SONGEVAL_PYTHON", sys.executable),
                       str(root / "scripts/songeval_worker.py"), "--repo", str(repo),
                       "--input", str(src), "--output", str(dst),
                       "--device", os.environ.get("MUSICBENCH_SONGEVAL_DEVICE", "auto")]
            try:
                proc = subprocess.run(command, capture_output=True, text=True,
                                      timeout=int(os.environ.get("MUSICBENCH_SONGEVAL_TIMEOUT", "3600")))
                if proc.returncode:
                    return {"available": False, "reason": proc.stderr[-2000:] or proc.stdout[-2000:]}
                result = json.loads(dst.read_text(encoding="utf-8"))
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                return {"available": False, "reason": str(exc)}
        per_sample = result["per_sample"]
        out = {"available": bool(per_sample), "n_evaluated": len(per_sample),
               "n_expected": len(rows), "coverage": len(per_sample) / len(rows),
               "per_sample": per_sample, "errors": result["errors"], "model": result["model"]}
        for dim in DIMENSIONS:
            values = [r[dim] for r in per_sample if r.get(dim) is not None and math.isfinite(r[dim])]
            out[dim] = sum(values) / len(values) if values else None
        return out
