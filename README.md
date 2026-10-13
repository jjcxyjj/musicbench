# MusicBench：只评测已有音频

现在只做一件事：**在已激活的 songeval 环境中，读取算法输出音频与 JSON 标注，选择评分能力，保存分数、图片和报告。** 不运行生成算法，不切换 conda 环境，不随机抽数据，不自动安装或下载模型。

## 代码放在哪里

```text
main.py                     # 读 JSON → 问可用输入 → 逐项评分
shared.py                   # 本地模型加载和分数汇总的小工具
config.yaml                 # 启用哪些能力、各自的本地模型路径
songeval/score.py            # 音频 → 五维美学评分
songeval/upstream/           # 你已有的 SongEval 仓库，也可在配置里填外部路径
audiobox-aesthetics/score.py # 音频 → CE/CU/PC/PQ
MuQ/score.py           # 音频 + 风格文本 → 相似度
CLAP/score.py                # 音频 + 风格文本 → CLAP 相似度
PER/score.py                 # ASR → PER / CER / WER
FAD/score.py                 # 生成集合 + GT 集合 → 分布距离
ACE/score.py                 # captioner 音乐描述 / transcriber 歌词识别
reporting/write.py          # CSV、JPG、Markdown 报告
examples/input.example.json # 输入示例
legacy/                     # 之前的复杂编排代码，仅归档，新入口不调用
```

各能力只有一个 `evaluate(rows, options, references)` 函数。代码库可放到对应目录的 upstream，权重放在 weights，也可在 config.yaml 填你已经跑通的外部绝对路径。不用把上游代码复制进评分文件。

## 第一步：先用已有的 SongEval 和 Audiobox

```bash
conda activate songeval   # 替换成你的实际环境名，只激活这一次
python main.py --input /data/generated/results.json
```

启动后会问：有没有目标歌词、GT/真实参考音频、风格描述。算法输出音频必须有。默认只运行 `enabled_metrics: [songeval, audiobox]`；其他能力会标为未启用，配好一个再把它的名字加进去。

模型路径在 config.yaml 改；不再为每个指标填环境名。所有能力在当前进程依次调用，每项结束后清理未引用对象与 CUDA 缓存。没有新的算法推理进程或 conda 子进程。

检查输入与计划但不加载模型：

```bash
python main.py --input /data/generated/results.json --plan-only
```

## JSON 怎么写

```json
[
  {
    "id": "song-1",
    "algorithm": "ace",
    "audio_path": "/data/generated/ace/song-1.wav",
    "has_vocal": true,
    "lyrics": "春风吹过山岗",
    "language": "zh",
    "style_prompt": "国风流行，钢琴与古筝，女声",
    "gt_audio_path": "/data/reference/song-1.wav"
  },
  {
    "id": "song-2",
    "algorithm": "ace",
    "audio_path": "/data/generated/ace/song-2.wav",
    "has_vocal": false,
    "style_prompt": "舒缓的古筝独奏",
    "gt_audio_path": "/data/reference/song-2.wav"
  }
]
```

- `audio_path`（或 generated_audio_path）必须指向**算法实际输出音频**。MusicLite 的原始 `path` 不自动当成生成结果，避免误评原始音频。
- 风格兼容 `style_prompt` / `Qwen_caption` / `caption`，不把描述当成歌词。
- `has_vocal: false` 的纯音乐不需要 lyrics。歌词评分只选 has_vocal=true 且有目标歌词的记录；未知人声标注或缺歌词的记录不参与歌词评分，仍参与美学评估。
- GT 兼容 gt_audio_path / path_original；也支持 `{"samples": [...], "reference_audio": ["a.wav", "b.wav"]}` 提供统一参考集合。选择有 GT 而 JSON 没有路径时，会问 GT 目录。
- algorithm 用于按算法分组比较，不启动该算法；不填则统一叫 generated。同一算法内 ID 必须唯一。
- 相对音频路径以输入 JSON 所在目录为基准；配置里的相对模型路径以 config.yaml 所在目录为基准。
- 原有或未来增加的标注会保留在 metadata 中。

## 输入决定哪些指标适用

| 能力 | 适用输入 | 是否默认启用 |
|---|---|---|
| SongEval | 算法输出音频 | 是 |
| Audiobox | 算法输出音频 | 是 |
| MuQ-MuLan | 算法输出音频 + 风格描述 | 配好后启用 muq_mulan |
| FAD | 每个算法的输出集合 + 真实参考集合 | 配好后启用 fad |
| CLAP | 算法输出音频 + 风格描述 | 配好后启用 clap |
| PER（Whisper） | 有人声 + 目标歌词 | 配好后启用 per |
| ACE captioner | 算法输出音频 → 音乐描述，不产生匹配分数 | 配好后启用 ace_captioner |
| ACE transcriber | 有人声 + 目标歌词 → PER/CER/WER | 配好后启用 ace_transcriber |

例如配好 MuQ 后修改 `enabled_metrics: [songeval, audiobox, muq_mulan, fad]`。也可临时 `python main.py --input results.json --metrics songeval audiobox`。

FAD 至少需要各两条生成/参考音频；样本过少仍不稳定。默认使用 **FAD-CLAP**。可设置 fad.embedding: muq_mulan 并在 fad 中填 MuQ-MuLan 的 model/audio_model/text_model 使用另一编码器。不同编码器、权重、预处理和参考集合的 FAD 不可直接比较。

## 各库仓库与本地模型

| 项目 | 官方仓库 | 本地模型 |
|---|---|---|
| SongEval | https://github.com/ASLP-lab/SongEval | 仓库 ckpt/model.safetensors + MuQ-large-msd-iter |
| Audiobox | https://github.com/facebookresearch/audiobox-aesthetics | checkpoint.pt |
| MuQ / MuQ-MuLan | https://github.com/tencent-ailab/MuQ | MuQ-MuLan-large + MuQ-large-msd-iter + xlm-roberta-base |
| FAD 工具参考 | https://github.com/microsoft/fadtk | FAD 是统计计算，默认复用 CLAP 特征，自行计算 Fréchet 距离，不额外依赖 fadtk |
| CLAP | https://github.com/LAION-AI/CLAP | 完整 Transformers 格式 CLAP |
| PER | https://github.com/SYSTRAN/faster-whisper | CTranslate2 格式 Whisper；中文 pypinyin / 英文 phonemizer |
| ACE 两个标注模型 | https://github.com/QwenLM/Qwen2.5-Omni | 各自完整的 captioner / transcriber 模型 |

联网准备机可执行以下命令，之后复制到服务器；服务器不运行下载命令：

```bash
git clone https://github.com/ASLP-lab/SongEval.git songeval/upstream
git clone https://github.com/facebookresearch/audiobox-aesthetics.git audiobox-aesthetics/upstream
git clone https://github.com/tencent-ailab/MuQ.git MuQ/upstream
git clone https://github.com/microsoft/fadtk.git FAD/upstream
git clone https://github.com/LAION-AI/CLAP.git CLAP/upstream
git clone https://github.com/SYSTRAN/faster-whisper.git PER/upstream
git clone https://github.com/QwenLM/Qwen2.5-Omni.git ACE/upstream
```

已有仓库不用克隆，把 config.yaml 的 repo 改成原路径即可。git clone 不保证包含权重；模型目录、分词器和权重文件需你手工准备。MuQ-MuLan 的音频/文本编码器分别填写 audio_model/text_model，本地目录名保留 muq / xlm-roberta 字样。

Audiobox 非 Hugging Face 官方权重地址：`https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt`。在联网机器下载，再放入 audiobox-aesthetics/weights/checkpoint.pt。

所有依赖在同一个 songeval 环境里手工配置。新入口只需 PyYAML；报告还需要 matplotlib；各 score.py 的第三方 import 就是该能力需要的库。已有 SongEval/Audiobox 不必重装。其他库可在同系统联网机器准备 wheelhouse，服务器用 `python -m pip install --no-index --find-links /data/wheelhouse 包名` 补齐。不会自动创建环境或更新版本。

## ACE 两个模型怎么运行

已拆成 ACE/acestep-captioner/run.py 和 ACE/acestep-transcriber/run.py，公用 ACE/inference.py，读取本地权重。官方没有在模型卡提供独立 GitHub 运行项目，而是明确沿用 Qwen2.5-Omni-7B 的加载与推理方式。

详见 [ACE 教程](ACE/README.md)，包含官方模型页、代码仓库、离线依赖和单音频命令。Captioner 输出描述，不充当风格匹配评分；Transcriber 从音频识别歌词，再通过 [PER 协议](PER/README.md) 计算错误率。它们不是 ACE 内部 LM PMI / DiT 注意力评分；旧 PMI 实验文件保留但不再由 main.py 调用。

## 输出

每次实验写入独立 results/时间目录：input.json、config.yaml、scores.json、scores.csv、每项可用指标的 JPG 和 report.md。部分能力失败会保留其他结果并非零退出；不适用、尚未启用分别标明。所有分数分项展示，不计算综合排名。

总体流程和文生音乐评测图保存在根目录的 workflow_overview.jpg、text2music_evaluation.jpg。旧代码已归档到 legacy，本 README 描述的是新的 main.py 入口。
