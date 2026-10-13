# 离线服务器：手动准备评测环境与本地模型

服务器无需联网。项目不再提供自动安装/克隆/下载：`scripts/setup_metrics.py` 现在只检查现有环境、依赖及本地权重，缺少时列出问题、代码来源和本教程。所有内置评分器只读取明确配置的本地文件；不会把缺失模型回退到 Hugging Face 下载。底层库可能使用 Hugging Face 的本地加载接口，但不要求服务器有 Hugging Face 缓存。

## 1. 目录约定

你可以把代码和模型放到各自库内，不需要统一缓存。下面是示例布局，路径均可在 YAML 中改为你现有目录：

```text
/data/musicbench-evaluators/
  audiobox-aesthetics/weights/checkpoint.pt
  SongEval/model.py, config.yaml, ckpt/model.safetensors
  MuQ/weights/
    muq/                 # MuQ 编码器完整本地模型目录
    mulan/               # MuQ-MuLan 完整本地模型目录
    xlm-roberta-base/    # 文本编码器、分词器完整目录
  CLAP/weights/clap/     # Transformers 格式 CLAP 模型和处理器
  faster-whisper/weights/large-v3/  # CTranslate2 格式 ASR
  demucs/weights/        # htdemucs.yaml + 对应 .th
```

模型可以来自你已有的库、离线备份或提供方支持的下载渠道。项目不会要求你使用 Hugging Face 下载器。**git clone 只获得代码，不保证包含预训练权重**；不要把任意 .pth 换文件名冒充另一种格式。下文列出了每个评分器需要的模型身份和格式，是否完整由本地加载检查确认。

## 2. 在能联网的准备机器上获取代码

以下命令在联网机器执行，然后把目录复制到服务器的对应位置。已有 SongEval 直接保留原仓库与原环境。

```bash
mkdir -p /data/musicbench-evaluators
cd /data/musicbench-evaluators
git clone https://github.com/facebookresearch/audiobox-aesthetics.git
git clone https://github.com/ASLP-lab/SongEval.git
git clone https://github.com/tencent-ailab/MuQ.git
git clone https://github.com/huggingface/transformers.git CLAP
git clone https://github.com/SYSTRAN/faster-whisper.git
git clone https://github.com/facebookresearch/demucs.git
```

CLAP 当前适配器使用 Transformers 的 ClapModel/ClapProcessor，代码仓库选的是 Transformers；LAION 原始 CLAP .pt 不能直接填进这个适配器。若你已有原始 CLAP 环境，需要提供兼容模型目录或另写对应指标适配器。

## 3. 逐项准备模型

| 指标 / 组件 | 本地配置项 | 需要准备的内容 | 代码及模型说明 |
| --- | --- | --- | --- |
| Audiobox CE/CU/PC/PQ | audiobox.checkpoint | 官方 checkpoint.pt 文件 | [Audiobox 官方仓库](https://github.com/facebookresearch/audiobox-aesthetics) |
| SongEval | songeval.repo、muq_model | 官方仓库内 ckpt/model.safetensors；MuQ-large-msd-iter 本地目录 | [SongEval](https://github.com/ASLP-lab/SongEval)、[MuQ](https://github.com/tencent-ailab/MuQ) |
| CLAP / FAD | clap_score.model、fad.model | laion/clap-htsat-fused 对应的 Transformers 完整模型目录 | [Transformers CLAP 文档](https://huggingface.co/docs/transformers/model_doc/clap)、[LAION CLAP 仓库](https://github.com/LAION-AI/CLAP) |
| MuQ-MuLan | mulan.model、audio_model、text_model | MuQ-MuLan-large 主模型、MuQ-large-msd-iter 音频模型、xlm-roberta-base 文本模型及分词器 | [MuQ 官方仓库](https://github.com/tencent-ailab/MuQ) |
| 歌词识别 | lyrics.model | faster-whisper large-v3 的 CTranslate2 完整模型目录 | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| 人声分离 | lyrics.separation.repo、model | Demucs htdemucs 的模型包 YAML 和 .th 文件 | [Demucs](https://github.com/facebookresearch/demucs) |

### Audiobox

联网机器可直接从官方非 Hugging Face 地址获取原始权重，然后拷贝到服务器：

```bash
mkdir -p /data/musicbench-evaluators/audiobox-aesthetics/weights
curl -L https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt \
  -o /data/musicbench-evaluators/audiobox-aesthetics/weights/checkpoint.pt
```

### SongEval 和 MuQ

SongEval 的打分 checkpoint 位于官方仓库 `ckpt/model.safetensors`。如果拷贝的是 Git LFS 指针，在联网准备机的仓库执行 `git lfs pull`，再转移实际文件。已有跑通版本不需要重新下载。

MuQ 本地目录需要 `config.json` 和原版兼容的 `model.safetensors`（或对应 PyTorch 权重）。SongEval 的 `muq_model` 指向此目录，不能填远端模型 ID。

MuQ-MuLan 不只是一个主权重：它的原配置还引用音频和文本编码器。YAML 中填写 audio_model/text_model 后，适配器在内存里改为本地路径，**不修改你库里的 config.json**。目录名分别保留 `muq` 和 `xlm-roberta` 字样，因为上游按这些字样选择模型类别。文本目录必须包含分词器文件，例如 tokenizer.json 或 sentencepiece.bpe.model 及相应 tokenizer_config.json。检查会执行一次短文本编码，发现未准备齐的依赖。

### CLAP / FAD

同一套本地目录可供两者复用，通常包含：config.json、model.safetensors 或 pytorch_model.bin、preprocessor_config.json、tokenizer_config.json，以及 tokenizer.json 或 vocab.json/merges.txt 等分词器文件。请保留模型提供方的完整导出目录，不能只拷贝权重。当前实验的 FAD 使用 CLAP 音频嵌入，不是其他编码器的 FAD。

### 歌词 ASR 与发音库

本地 ASR 目录需要 config.json、model.bin、tokenizer.json 和模型对应的其他配置/词表。原始 OpenAI Whisper .pt 不是 faster-whisper 可直接读取的格式。已有兼容 CTranslate2 导出可直接使用；如需转换，在联网准备机按 [faster-whisper 转换说明](https://github.com/SYSTRAN/faster-whisper#model-conversion) 完成后转移。请选与当前库版本匹配的模型。

歌词环境还需要 pypinyin、phonemizer 和 espeak-ng。espeak-ng 是动态库，不仅仅是 Python 包。检查程序会在该环境 lib 目录查找；若系统安装在其他位置，在 YAML 的 `metric_options.lyrics.environment.PHONEMIZER_ESPEAK_LIBRARY` 中填写实际 .so 路径。

### Demucs

默认 htdemucs 需要同目录的 `htdemucs.yaml` 和 `955717e8-8726e21a.th`。联网准备机可从官方非 Hugging Face 地址获取：

```bash
mkdir -p /data/musicbench-evaluators/demucs/weights
cp /data/musicbench-evaluators/demucs/demucs/remote/htdemucs.yaml \
  /data/musicbench-evaluators/demucs/weights/
curl -L https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th \
  -o /data/musicbench-evaluators/demucs/weights/955717e8-8726e21a.th
```

当前使用 `--repo 本地目录`，不会查询远端仓库。换成其他 Demucs 模型时，需准备其 YAML 中列出的全部模型文件。选择混音直接识别时，把 `lyrics.audio_source` 设为 `mix`，无需 Demucs。

## 4. 手动准备 conda 环境

最稳妥的是在与服务器相同系统、架构且兼容驱动的联网 Linux 机器准备环境，再打包迁移。不能把 macOS 环境直接复制到 Linux。以下命令均在联网准备机项目根目录执行：

```bash
conda env create -f envs/benchmark.yml
conda env create -f envs/metric-audiobox.yml -n musicbench-eval-audiobox
conda env create -f envs/metric-clap.yml -n musicbench-eval-clap
conda env create -f envs/metric-mulan.yml -n musicbench-eval-mulan
conda env create -f envs/metric-lyrics.yml -n musicbench-eval-lyrics
conda env create -f envs/metric-demucs.yml -n musicbench-eval-demucs
# 仅在没有现成 SongEval 环境时创建
conda env create -f envs/songeval.yml -n musicbench-eval-songeval
```

依赖对应关系：audiobox_aesthetics→Audiobox；transformers/librosa/torch→CLAP；muq→MuQ/MuLan；faster_whisper/pypinyin/phonemizer→歌词；demucs→分离；omegaconf/safetensors/einops→SongEval。模板还包含 ffmpeg；歌词环境包含 espeak-ng。检查器会列出缺少或无法导入的库及其异常原因。

可用 [conda-pack 官方工具](https://conda.github.io/conda-pack/) 迁移，例如：

```bash
# 联网准备机
conda install -n base -c conda-forge conda-pack
conda pack -n musicbench-eval-clap -o musicbench-eval-clap.tar.gz

# 离线服务器：解包到最终路径，只在该环境里运行 conda-unpack
mkdir -p /data/conda/envs/musicbench-eval-clap
tar -xzf musicbench-eval-clap.tar.gz -C /data/conda/envs/musicbench-eval-clap
/data/conda/envs/musicbench-eval-clap/bin/conda-unpack
```

其他环境同样处理。YAML 可直接填 `/data/conda/envs/musicbench-eval-clap` 这种绝对前缀路径，无需强求 conda 名称登记。主环境也要迁移；MusicBench 代码在服务器主环境中可以 `pip install --no-deps --no-build-isolation -e .`，前提是主环境的依赖和 setuptools/wheel 已准备好。

若选择 wheelhouse 手工安装，需在匹配服务器的联网机器下载所有依赖和构建依赖，离线用 `pip install --no-index --find-links /data/wheelhouse ...`。Git 仓库的可编辑安装也需要准备构建工具，不能让 pip 在隔离构建阶段偷偷尝试联网。CUDA 版 PyTorch 和 faster-whisper/CTranslate2 的运行库分别核对；CPU 加载通过不等于 CUDA 已验证。

## 5. 填 YAML 并离线检查

```bash
conda activate musicbench
python scripts/setup_metrics.py --init-config
# 手动编辑 configs/metrics.local.yaml 中的环境、仓库和权重路径
python scripts/setup_metrics.py --libraries-only
python scripts/setup_metrics.py
./run.sh text2music
```

`--init-config` 只复制 [本地配置样例](../configs/metrics.offline.example.yaml)，已有文件时拒绝覆盖。默认检查库和本地权重能否在 CPU 加载；`--libraries-only` 只检查依赖，不代表权重准备完成；`--check-only` 保留为兼容别名，所有模式都离线。`--metrics audiobox songeval clap_score fad` 可仅检查当前需要的组件。`--dry-run` 只显示配置。

缺环境时打印环境模板；缺库时列出导入失败的库；缺模型时打印 YAML 配置项和本地路径。诊断保存至 `work/metric-check/<时间>/status.json` 与各组件日志，**不会安装环境、git clone、下载权重或覆盖原配置**。检查成功后可先跑 1–2 条记录确认服务器 CUDA。

模型来源格式参考上游官方说明；本项目不保证提供方对每个模型均有非 Hugging Face 镜像。你可以使用已有本地权重，但须保持当前评分协议对应的模型与格式，替换模型要在报告中记录，不能把不同编码器结果混为同一个分数。
