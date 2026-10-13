# 评测库、模型与环境配置

主环境只编排实验。评分库及模型放进各自的 conda 环境，通过 `conda run` 串行调用，不需要手动反复 activate。生成算法的依赖仍由算法自己的环境负责。

## 1. 准备主环境

在项目根目录执行：

```bash
conda env create -f envs/benchmark.yml
conda activate musicbench
pip install -e .
python scripts/setup_metrics.py --dry-run
```

最后一条只显示计划，不创建环境、不下载、不修改配置。后续安装命令也在主环境执行。

## 2. 准备全部主指标

已有跑通的 SongEval 时，替换下面环境名和路径：

```bash
python scripts/setup_metrics.py \
  --reuse songeval=your_songeval_env \
  --songeval-repo /data/SongEval \
  --cache-dir /data/musicbench-models
```

没有现成 SongEval 时，直接运行 `python scripts/setup_metrics.py`。脚本创建缺少的评分环境，缺少 SongEval 仓库时从官方仓库克隆。已有仓库不会被修改或拉取更新。权重如果是 Git LFS 指针，会明确要求在仓库中执行 `git lfs pull`。

默认按顺序准备下表内容；CLAP 与 FAD 去重，不重复安装。同一时刻只运行一个安装/加载进程。

| 评测内容 | 默认 conda 环境 | 依赖与权重 |
| --- | --- | --- |
| Audiobox CE/CU/PC/PQ | musicbench-eval-audiobox | audiobox_aesthetics；官方默认 checkpoint |
| SongEval 五维评分 | musicbench-eval-songeval | 官方仓库、ckpt/model.safetensors、OpenMuQ/MuQ-large-msd-iter |
| 风格 CLAP + 分布 FAD | musicbench-eval-clap | Transformers CLAP；laion/clap-htsat-fused（模型和处理器） |
| 风格 MuQ-MuLan | musicbench-eval-mulan | muq；OpenMuQ/MuQ-MuLan-large 及其文本编码器/分词器 |
| 歌词 PER/CER/WER | musicbench-eval-lyrics | faster-whisper large-v3；pypinyin；phonemizer；espeak-ng |
| 歌词识别前的人声分离 | musicbench-eval-demucs | Demucs htdemucs |

PER/CER/WER 和 FAD 的统计计算代码已经在项目中，不需要另外下载“PER 模型”或“FAD 模型”。需要下载的是 ASR、人声分离和用于提取音频特征的模型。英文 PER 还依赖系统动态库 espeak-ng，环境模板会安装它，配置脚本会检测并记录库路径。

Hugging Face 与 Torch 权重默认缓存在项目 `models/metrics/`；`--cache-dir` 可改为服务器大容量目录。已有 `HF_HOME`/`TORCH_HOME` 会被尊重，显式 `--cache-dir` 则优先使用指定目录。缓存、日志和本机 YAML 已加入 Git 忽略列表，不会提交权重。

成功后生成 `configs/metrics.local.yaml`，包含环境、模型、共享缓存路径及检查记录。默认会真正加载权重；MuQ-MuLan 还检查文本编码以准备依赖模型。检查使用 CPU，避免占用生成算法的 CUDA 显存；这不等于已验证服务器上的 CUDA 推理。主指标真实评测时建议先用 1–2 条样本确认。

## 3. 复用环境、选择安装内容

```bash
# clap_score 和 fad 共用一个环境，可复用现有环境
python scripts/setup_metrics.py --reuse clap=my_clap_env

# 环境路径也支持（绝对路径）
python scripts/setup_metrics.py --reuse lyrics=/data/conda/envs/asr

# 仅准备美学和 CLAP/FAD；之后可再运行完整安装补齐
python scripts/setup_metrics.py --metrics audiobox songeval clap_score fad

# 歌词使用混音识别，跳过 Demucs；所有算法采用同一个协议
python scripts/setup_metrics.py --metrics lyrics --no-separation

# 较小 ASR 模型用于环境试运行；正式比较固定相同模型
python scripts/setup_metrics.py --metrics lyrics --whisper-model small

# 改用一组新的环境名，不覆盖原环境
python scripts/setup_metrics.py --env-prefix trial-eval- --output configs/trial-metrics.yaml
```

`--reuse` 左侧为组件：audiobox / songeval / clap / mulan / lyrics / demucs；右侧为已有 conda 环境名或绝对前缀路径。显式指定的复用环境不存在时直接报错。已存在环境仅做检查，不自动安装、覆盖或升级库；缺库请在该环境中自行补齐，或选一个新的环境名前缀创建隔离环境。

再次运行时会读取原 YAML，并保留自定义模型选项；显式命令行的环境、缓存、SongEval 路径和 Whisper 模型覆盖相应旧值。部分指标安装不会关闭任务规定的其他指标，完整实验中仍需安装实际启用的全部评分器。

## 4. 只装库、离线检查、离线服务器

```bash
# 创建缺少的环境，只检查库导入及歌词发音库，不下载/检查模型
python scripts/setup_metrics.py --libraries-only

# 已安装后，离线检查库与模型是否能加载；不安装、不克隆
python scripts/setup_metrics.py --check-only

# conda 不在 PATH 中时
python scripts/setup_metrics.py --conda /opt/miniconda3/bin/conda
```

`--libraries-only` 会生成标记为未验证权重的 YAML；正式实验前仍须准备模型。`--check-only` 禁用 Hugging Face 在线访问，并阻止 Torch 权重下载；缺模型会失败，不会自动转为在线下载。

离线服务器需要先在联网机器准备相同系统/架构的 conda 环境、模型缓存和 SongEval 仓库，再搬运到服务器。在服务器按实际环境名和缓存路径重新执行配置检查；不要直接照搬其他机器的绝对路径。只复制本项目代码不会带上权重。

## 5. 开始实验

```bash
./run.sh text2music

# 指定另一份评分器配置
./run.sh text2music --metric-config configs/trial-metrics.yaml

# 已有实验 YAML：其显式配置会覆盖评分器默认值
./run.sh --config configs/my-experiment.yaml
```

交互界面的默认环境和模型来自评分器 YAML，仍会询问设备、显卡、歌词语言与测试参数。保存实验时会把实际使用的评分器配置写入实验 YAML，后续复现实验不依赖交互答案。只有提供风格描述才调风格指标；只有提供目标歌词才调歌词指标；只有提供参考音频集合才计算 FAD。

## 6. 失败与版本兼容

各组件日志与 JSON 检查结果在 `work/metric-setup/<时间>/`；某项失败会继续检查其他组件，最后非零退出，并保留原本可用的配置，不发布半成品 YAML。修复后重跑即可，成功创建的环境和缓存会复用。

- SongEval：使用你已经跑通的环境和仓库最直接；检查 `model.py`、`config.yaml`、`ckpt/model.safetensors` 是否存在。
- 权重下载失败：确认服务器能访问对应上游；可先在联网机器准备缓存后迁移。
- 英文发音库失败：确认 lyrics 环境存在 espeak-ng；脚本会尝试在该环境 lib 目录定位动态库，也支持预设 `PHONEMIZER_ESPEAK_LIBRARY`。
- CUDA 推理失败：CPU 加载检查通过不能证明 CUDA 兼容。PyTorch 与 faster-whisper/CTranslate2 的 CUDA 依赖不同；按服务器驱动及上游说明选择兼容版本，或先在交互界面选 cpu 验证流程。不要为了评分器修改生成算法的环境。

上游配置来源：[Audiobox](https://github.com/facebookresearch/audiobox-aesthetics)、[SongEval](https://github.com/ASLP-lab/SongEval)、[MuQ/MuLan](https://github.com/tencent-ailab/MuQ)、[CLAP](https://huggingface.co/laion/clap-htsat-fused)、[faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[Demucs](https://github.com/facebookresearch/demucs)、[phonemizer](https://bootphon.github.io/phonemizer/install.html)。依赖模板是安装起点；实际安装版本记录在检查结果和实验产物中。
