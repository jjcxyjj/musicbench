# 怎么接入你的 text-to-music 模型

musicbench 只做三件事：**打包数据 → 跑你的模型 → 打分排名**。你唯一要写的，是
**一个推理脚本 `infer.py`**。它和 benchmark 之间只靠两个 JSON 文件通信，互不认识
对方的依赖：

```
manifest.json  ──(读)──>  你的 infer.py  ──(写)──>  predictions.jsonl
  输入: caption + 参考音频        生成音频                 output: audio_path
```

## 1. 你的模型需要做什么（唯一要写的代码）

`infer.py` 的逻辑只有三步：

```python
import json

with open(args.manifest) as f:
    manifest = json.load(f)          # 1) 读输入

for row in manifest:
    caption = row["metadata"]["caption"]   # 2) 取文本提示词
    audio = your_model.generate(caption)   #    ← 这里调你的模型
    save(audio, f"{row['id']}.wav")        #    存成 wav

    write_jsonl({"id": row["id"],
                 "output": {"audio_path": "/path/to/saved.wav"}})  # 3) 写输出
```

**一句话：`metadata.caption` 是输入，`output.audio_path` 是输出。** 中间那一行
换成你真实的生成调用即可。

## 2. manifest.json 里有什么

打包后每一条长这样（字段来自你的 `5_with_latent.json`）：

```json
{
  "id": "ml_chunk3_0",
  "audio_path": "audio/ml_chunk3_0.wav",     // 参考音频（打分用，你不用读它）
  "metadata": {
    "caption": "The music is an instrumental ...",   // ← 你的输入 prompt
    "has_vocal": true,
    "vocal_score": 0.21,
    "style_name": "国风流行",
    "style_code": "gflx"
  },
  "ground_truth": {"caption": "...", "has_vocal": true}
}
```

## 3. predictions.jsonl 里你要写什么

每行一个 JSON，`id` 必须和 manifest 对得上：

```json
{"id": "ml_chunk3_0", "output": {"audio_path": "/abs/path/gen/ml_chunk3_0.wav"}}
```

`audio_path` 用**绝对路径**最稳妥（打分脚本在另一个 conda 环境里跑，相对路径会错位）。

## 4. 打分是怎么算的（你不用管，知道即可）

benchmark 拿你的 `audio_path` 和 manifest 里的参考音频 + caption 对比：

- **basic 档**：客观特征 MAE（时长/响度/频谱质心/过零率/crest/静音占比），越接近参考越好。
- **professional 档**：FAD（分布距离）、KL（标签分布）、CLAP（文本对齐）。

最后多指标归一化加权 → 排名出「谁最好」。

## 5. 目录结构约定

```
algorithms/<你的模型名>/
├── config.yaml          # name / env / task / infer_script
└── infer.py             # 你要写的推理脚本
```

`config.yaml` 里 `env:` 填你模型自己的 conda 环境名（每个算法各一个，互不干扰）：

```yaml
name: my_t2m
env: my_t2m_env        # ← 你模型自己的 conda 环境
task: text2music
infer_script: infer.py
```

## 6. 跑起来

```bash
./run.sh text2music
```

它会：打包数据（默认随机抽 10 条）→ 对每个 `task: text2music` 的算法，用**它自己
config.yaml 里的 env** 跑 `infer.py` → 打分 → 出排名。
