"""t2m_passthrough — a placeholder text-to-music algorithm.

It does NOT actually generate; it just copies the reference audio as the
"generated" output. This is intentionally useless in practice, but it lets the
benchmark pipeline run end-to-end and gives a perfect-score baseline for the
L0 objective metrics (identical audio => MAE 0).

Replace this with your real model: read the caption from manifest metadata,
generate audio, and write each record's ``output.audio_path`` to a wav file.

    python infer.py --manifest dataset/manifest.json --output preds/t2m_passthrough.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, List


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--out-audio-dir", default=None,
                   help="Directory to write generated audio (default: alongside output)")
    args = p.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    base = os.path.dirname(args.manifest)
    out_dir = args.out_audio_dir or os.path.join(os.path.dirname(args.output), "gen_audio")
    os.makedirs(out_dir, exist_ok=True)

    lines: List[str] = []
    for row in manifest:
        ref_path = row.get("audio_path", "")
        if not ref_path or not os.path.isabs(ref_path):
            ref_path = os.path.join(base, ref_path) if ref_path else ""

        gen_path = ""
        if ref_path and os.path.isfile(ref_path):
            ext = os.path.splitext(ref_path)[1] or ".wav"
            gen_path = os.path.join(out_dir, f"{row['id']}{ext}")
            shutil.copy2(ref_path, gen_path)

        out: Dict[str, Any] = {
            "id": row["id"],
            "output": {"audio_path": gen_path, "caption_used": (row.get("metadata") or {}).get("caption", "")},
        }
        lines.append(json.dumps(out, ensure_ascii=False))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[t2m_passthrough] wrote {len(lines)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
