"""真实 text2music 推理模板（本地 Python 权重形态）。

照抄这个文件，只需要改 `load()` 和 `generate()` 两个函数。

    python infer.py --manifest dataset/manifest.json --output preds/my_t2m.jsonl

约定：
  * 输入：manifest 每条 `metadata.caption`（文本 prompt）
  * 输出：predictions.jsonl 每行 `{"id": ..., "output": {"audio_path": "绝对路径"}}`
  * 本脚本不 import musicbench（算法环境不需要装 musicbench）
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

# =============================================================================
# TODO: 换成你模型的 import，例如：
#   from your_model import YourModel
#   import torch
# =============================================================================


class YourModel:
    """占位类 —— 换成你的真实模型。"""

    def load(self, checkpoint_path: str) -> None:
        # TODO: 加载权重，例如：
        #   self.model = YourModel.from_pretrained(checkpoint_path)
        #   self.model.eval()
        print(f"[my_t2m] loading checkpoint from {checkpoint_path}")

    def generate(self, caption: str, sr: int = 48000) -> "object":
        """根据 caption 生成音频，返回 (numpy array, sample_rate)。

        TODO: 换成你的真实生成调用，例如：
            audio = self.model.generate(prompt=caption)
        """
        # 下面是占位实现（生成 3 秒 440Hz 正弦），务必替换掉。
        import numpy as np

        t = np.arange(sr * 3) / sr
        return (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32), sr


# =============================================================================
# 下面的代码通常不用改
# =============================================================================


def save_wav(path: str, audio: "object", sr: int) -> None:
    from scipy.io import wavfile

    import numpy as np

    wavfile.write(path, sr, np.asarray(audio, dtype=np.float32))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default=os.environ.get("MY_T2M_CHECKPOINT", ""),
                   help="模型权重路径（或读环境变量 MY_T2M_CHECKPOINT）")
    p.add_argument("--sr", type=int, default=48000)
    args = p.parse_args()

    # 1. 加载模型
    model = YourModel()
    model.load(args.checkpoint)

    # 2. 读输入
    with open(args.manifest) as f:
        manifest = json.load(f)

    # 3. 生成 + 写输出（每个算法独立目录，避免多算法互相覆盖）
    out_dir = os.path.join(os.path.dirname(args.output) or ".", "gen_audio_my_t2m")
    os.makedirs(out_dir, exist_ok=True)

    lines: List[str] = []
    for row in manifest:
        caption = (row.get("metadata") or {}).get("caption") or ""
        if not caption:
            print(f"[my_t2m] skip {row['id']}: no caption")
            continue

        audio, sr = model.generate(caption, sr=args.sr)
        audio_path = os.path.abspath(os.path.join(out_dir, f"{row['id']}.wav"))
        save_wav(audio_path, audio, sr)

        out: Dict[str, Any] = {"id": row["id"], "output": {"audio_path": audio_path}}
        lines.append(json.dumps(out, ensure_ascii=False))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[my_t2m] wrote {len(lines)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
