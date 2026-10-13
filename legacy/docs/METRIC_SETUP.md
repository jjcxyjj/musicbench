# 离线服务器：手动准备评测环境与本地模型

服务器无需联网。项目不再提供自动安装/克隆/下载：`scripts/setup_metrics.py` 现在只检查现有环境、依赖及本地权重，缺少时列出问题、代码来源和本教程。所有内置评分器只读取明确配置的本地文件；不会把缺失模型回退到 Hugging Face 下载。底层库可能使用 Hugging Face 的本地加载接口，但不要求服务器有 Hugging Face 缓存。

## 优先复用一个评分环境

默认所有评分器共用一个 conda 环境，模型和权重仍分别放在各自库里。优先复用你已跑通的 SongEval，也可以把评分依赖放进 musicbench：只需修改配置中的一行。

```yaml
evaluation_environment: songeval  # 替换为你的实际 SongEval 环境名，也可填 musicbench
```

先检查现有环境，不安装任何内容：

```bash
python scripts/setup_metrics.py --shared-env songeval --libraries-only
# 或检查 musicbench
python scripts/setup_metrics.py --shared-env musicbench --libraries-only
```

以上命令临时强制所有评分器用同一环境，原 YAML 不变。缺库时按下面教程在该环境手工补齐，已装且兼容的库不必重复安装。然后用同一命令去掉 --libraries-only 检查本地模型。检查通过后将 evaluation_environment 改为实际环境名。

新配置也可一次生成：`python scripts/setup_metrics.py --init-config --shared-env songeval`。已有配置不用重建；实验运行时可以 `./run.sh text2music --evaluation-env songeval` 强制统一全部评分器（含 Demucs）。交互界面默认只问一次评分环境，选择“为个别评分器单独指定环境”时才展开独立环境设置。

如果确有冲突，可只覆盖该项，例如 `metric_options.lyrics.separation.env: demucs_env`，其他指标仍共用。环境优先级：指标 options.env > metric_environments 映射 > evaluation_environment > current；命令行 --evaluation-env 显式选择共用环境时会清除各项环境覆盖。生成算法的 env 不受影响。

依赖合并的候选模板为 [envs/evaluation.yml](../envs/evaluation.yml)，包含主框架与所有评分库。模板按 SongEval 的 torch/torchaudio 2.7.0、MuQ 0.1.0、Audiobox 0.0.4、Demucs 4.0.1 等版本组织，目标是 Linux 服务器；包声明的约束可重叠，并不等于已验证你的 CUDA、动态库和模型。不要用模板覆盖已跑通的 SongEval；优先对原环境检查并补缺库。

本机只有轻量 musicbench 环境，评分库尚未安装，也没有你的服务器 SongEval 环境；因此没有宣称全部模型已在一个环境跑通。macOS 的依赖解析也不能替代 Linux 服务器验证。

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

优先在已跑通的 SongEval 环境补齐缺少的评分库，或在 musicbench 中准备全部依赖；算法环境继续独立。联网准备机应与服务器同系统、架构和兼容驱动，不能把 macOS 环境直接复制到 Linux。

如果没有可复用的评分环境，可在联网 Linux 准备机只创建一套完整环境：

```bash
conda env create -f envs/evaluation.yml
# 这套环境也包含主框架依赖，可同时承担编排与评分
conda activate musicbench-eval
pip install --no-deps --no-build-isolation -e .
# 配置 evaluation_environment: musicbench-eval
# 启动主流程：BENCH_ENV=musicbench-eval ./run.sh text2music
```

已有 SongEval/musicbench 环境时，用已准备好的离线 wheelhouse 手工补缺，例如在服务器激活目标环境后 `python -m pip install --no-index --find-links /data/wheelhouse audiobox-aesthetics faster-whisper pypinyin phonemizer`。wheelhouse 应包含所需版本和全部依赖；PyTorch/torchaudio 保持匹配，避免无意升级已跑通的版本。ffmpeg、espeak-ng 等系统组件也须放进该环境或准备动态库路径。

逐项 envs/metric-*.yml 保留作为确实冲突时的备用模板，不再默认创建六套环境。Demucs 固定 4.0.1 是为了保持当前 get_model/--repo 接口和评分协议；更新大版本需先核对 API。官方本地 Demucs 4.0.1 checkpoint 包含模型类，适配器仅在分离/检查子进程启用与 PyTorch 2.7 的加载兼容设置，不改变其他评分器进程。

可用 [conda-pack 官方工具](https://conda.github.io/conda-pack/) 迁移，例如：

```bash
# 联网准备机
conda install -n base -c conda-forge conda-pack
conda pack -n musicbench-eval -o musicbench-eval.tar.gz

# 离线服务器：解包到最终路径，只在该环境里运行 conda-unpack
mkdir -p /data/conda/envs/musicbench-eval
tar -xzf musicbench-eval.tar.gz -C /data/conda/envs/musicbench-eval
/data/conda/envs/musicbench-eval/bin/conda-unpack
```

复用 SongEval 时打包你的实际环境名即可。YAML 的 evaluation_environment 可直接填 `/data/conda/envs/musicbench-eval` 这种绝对前缀路径，无需强求 conda 名称登记。主环境也要迁移；MusicBench 代码在服务器主环境中可以 `pip install --no-deps --no-build-isolation -e .`，前提是主环境的依赖和 setuptools/wheel 已准备好。

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
