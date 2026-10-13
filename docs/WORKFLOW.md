# 文生音乐主指标工作流

入口不区分简单/复杂，也不要求用户逐项挑选指标。任务在 `configs/tasks.yaml` 定义固定指标及输入条件，交互只确认可用输入、算法、参数、数据和评分环境。

```bash
./run.sh
./run.sh text2music
./run.sh --dry-run --save-config configs/my-experiment.yaml
./run.sh --config configs/my-experiment.yaml --output runs/experiment-01
# 已激活主环境
BENCH_ENV=current ./run.sh
```

输出目录必须不存在，避免覆盖实验。`--dry-run` 只解析和校验配置，不检查全部依赖或加载权重。文件路径相对于实验 YAML 所在目录，infer_script 相对于算法 cwd；交互输入路径相对于项目根目录。环境名也可填绝对环境路径，使用 conda `-p` 运行；current 表示主进程 Python。

## 固定指标与分类

- 美学（含结构）：Audiobox CE/CU/PC/PQ；SongEval coherence/musicality/memorability/clarity/naturalness。
- 歌词匹配：单次音频识别后，计算 PER 与中文 CER / 英文 WER，保存替换、删除、插入对齐明细。
- 风格匹配：CLAP、MuQ-MuLan，输入仅 style_prompt，不拼接 lyrics。
- 分布距离：FAD，当前使用 CLAP 音频特征，不与使用其他编码器的 FAD 混比。

PC 是描述性复杂度，不预设高分更好；生成任务不输出综合排名。SongEval naturalness 针对人声呼吸与分句，仅 has_vocal=true 的样本参与；无明确人声标注显示不适用。

CLAP 与 MuQ-MuLan 采用官方接口处理整段音频；模型本身可能裁剪或池化长音频，不能解释为逐秒、逐乐器的完整条件验证。FAD 对整个生成集合与固定参考集合计算，至少需要各两条音频；小样本结果不稳定。FAD 不要求逐条配对，也不等于多样性或防复制指标。

## 输入数据

推荐 `manifest` JSON 数组。可以只提供文本，没有参考音频也可生成和评测。字段可放顶层或 metadata；metadata 优先。caption 向后兼容映射为 style_prompt，不映射为歌词。

```json
[
  {"id":"sample-1","style_prompt":"warm acoustic pop, guitar, female vocals","lyrics":"[Verse]\n你好世界","language":"zh","has_vocal":true},
  {"id":"sample-2","style_prompt":"instrumental piano solo","has_vocal":false}
]
```

如有音频，增加 audio_path，相对路径以原始 manifest 所在目录为基准。程序复制已提供音频并保存实际测试清单；缺失音频文件报错。任务要求真实标签时提供 ground_truth，例如节拍任务的 beats/tempo，标签任务的 tags。

MusicLite `feature_json` 也支持：读取 root 下各子目录的 5_with_latent.json，可配置 caption_field、lyrics_field、language_field。旧 Qwen_caption 仍用作风格描述。manifest 请求数量大于记录数会报错；feature_json 沿用请求超出时返回全部记录的行为，报告使用实际数量。

```yaml
inputs:
  style_prompt: true
  lyrics: true
  reference_audio: false
```

true 表示声明每条抽样记录都有对应输入，缺失会报错；false 显式禁用对应条件评测；不写则根据样本字段选择适用子集。混合数据集可省略 style_prompt/lyrics 标志，仅对有对应字段的样本评分，记录适用样本 ID 与数量。不提供输入的指标标为 not_applicable，不伪造零分。

FAD 的参考集合需要显式声明；不会因 manifest 恰好有音频就自动启用：

```yaml
inputs: {reference_audio: true}
reference_audio:
  source: directory
  path: /data/real_music
  num_samples: 100  # 0 表示全部；固定种子抽样
# 或 reference_audio: {source: dataset} 使用本次抽样的 audio_path
```

参考路径保存在 reference_manifest.json，生成样本只抽一次。repetitions>1 时，清单中每条输入扩展为唯一输出 ID，并包含 source_id、repetition、generation_seed；全部输出参与评分，不做 best-of-N 筛选。真实算法应读取 generation_seed 并设置自身框架随机种子。

## 算法配置与参数

每个算法的 config.yaml 声明：

```yaml
name: your_generator
task: text2music
env: your_generator_env
infer_script: infer.py
parameter_schema:
  checkpoint: {flag: --checkpoint, type: path, required: true, label: 模型权重路径}
  steps: {flag: --steps, type: int, minimum: 1, default: 30, label: 采样步数}
  guidance: {flag: --guidance, type: float, default: 3.0, label: 引导强度}
parameters: {}
```

界面根据 schema 提问并保存 parameters。类型支持 str/int/float/path/bool，可配置 default、required、minimum、choices。bool 使用“为真则传 flag”的形式；需要 `--flag true/false` 的算法应定义字符串 choices。没有 schema 时可以输入额外参数，解析为 args 列表；YAML 的 args 必须是字符串列表。

每个算法可设置 cuda_visible_devices: '0' 或 '1'；inherit 继承外部配置，cpu 将 CUDA_VISIBLE_DEVICES 设为空。工作进程里的 cuda:0 对应所选物理卡。模型、生成时长等不强制统一，由适配脚本声明参数，但比较报告应注明条件不同。提供的 my_t2m 仍是正弦音占位模板，不是真实生成模型。

推理契约：读取 `--manifest ABS_PATH`，写 `--output ABS_PATH`；预测 JSONL 每行为 `{"id":"sample-1","output":{"audio_path":"/abs/generated.wav"}}`。预测音频相对路径以 JSONL 所在目录为基准。必须覆盖全部样本且 ID 唯一。每个算法输出独立目录，所有推理及评测器串行运行，超时会终止该进程组。

## 评测环境

主环境只安装 MusicBench 基础依赖。可按下表自行准备独立环境，也可以复用已有环境；库和权重安装由用户管理。

| 评分器 | 主要依赖 | 模型配置 |
| --- | --- | --- |
| Audiobox | audiobox_aesthetics / torch | checkpoint 留空用官方默认；也可指定 checkpoint.pt |
| SongEval | 原项目依赖 | repo 指向已跑通仓库；env 填原有环境 |
| CLAP / FAD | torch / transformers / librosa | 默认 laion/clap-htsat-fused |
| MuQ-MuLan | muq / torch / librosa | 默认 OpenMuQ/MuQ-MuLan-large |
| 歌词 | faster-whisper / pypinyin / phonemizer | 默认 large-v3，可填本地 faster-whisper 模型路径 |
| 人声分离 | demucs / torch | 默认 htdemucs，可独立环境 |

对应模板为 envs/metric-*.yml。模板不会自动创建环境或安装权重。CLAP、MuLan 和 Audiobox 可按库行为下载官方权重；无网络运行请预先缓存或使用本地权重。GPU 版 PyTorch/CTranslate2 需与实际 CUDA/驱动兼容。

```yaml
metric_environments: {audiobox: audiobox, lyrics: lyrics, clap_score: clap, mulan: mulan, fad: clap}
metric_options:
  songeval: {repo: /path/to/SongEval, env: your_songeval_env, device: auto, cuda_visible_devices: '0'}
  lyrics:
    model: large-v3
    device: auto
    language: zh
    beam_size: 5
    audio_source: demucs
    separation: {env: demucs, device: auto, model: htdemucs, cuda_visible_devices: '0'}
```

其他评测器的 metric_options 同样可设 env、device、cuda_visible_devices、timeout；model 表示官方标识或本地模型目录。Audiobox 官方库自动选择设备，指定 device 后会将模型迁移至该设备。

## 歌词协议

默认所有算法都从最终混音用 Demucs 提取人声，避免因有些算法提供原生干声而处理条件不同。也可统一设置 audio_source: mix 直接识别整曲；不自动在两种协议间切换。分离进程退出后才启动识别进程。

ASR 只接收音频与 language/beam_size 等解码设置，不接收目标歌词，不用目标歌词提示识别。逐样本 metadata.language 优先于 metric_options.lyrics.language；当前主协议支持 zh/en，其他语言明确报错。

规范化：Unicode NFKC、小写化、去除 []/<> 结构/表演标签与标点。中文计算 CER，英文计算 WER；二者分别统计，不混成一个词错率。

PER：中文采用 pypinyin strict 声母与带声调的韵母（中性调为 5），英文采用 phonemizer 的 espeak en-us 音素、无重音。英文环境需系统 espeak-ng 共享库，模板通过 conda-forge 安装；若 phonemizer 找不到库，可配置 PHONEMIZER_ESPEAK_LIBRARY 指向环境中的 lib/libespeak-ng.so（macOS 对应 dylib）。当前中文 PER 对混合拉丁文字和数字明确报错，保留 CER 与转写，而不是悄悄将未转换文字当成音素。

各错误率按参考单位数量微平均；重复、多唱对应插入，漏唱对应删除，唱错对应替换，但都属于 ASR/G2P 代理判断。识别/分离错误会影响指标；空转写按全删处理。G2P 失败时保留词/字错率并标记 PER 失败。不将 PER 解释为时间同步评分；不声称等价 ACE 内部 AAS 或专有 ACE-Lyric-Reward。

## 输出、失败和扩展

输出包括 config.yaml、status.yaml、evaluation_plan.yaml、dataset/manifest.json、reference_manifest.json、predictions、metric_results、logs，以及 results/scores.json、env.json、comparison.png、分类 PNG、report.md/report.log。指标原始结果、模型版本、转写、音素和差异明细会保留。

一个评测器失败会继续收集其他结果和其他算法，最后生成带错误说明的报告。默认 require_metrics=true，所有应该运行的指标必须完整，否则报告保留、状态 failed、进程非零退出；不适用不算失败。算法推理失败也会记录，并在文生音乐任务继续下一算法。

新增任务：注册 Task，在 tasks.yaml 增加 dataset 与 metrics；新增指标：注册 Metric，通过 plugins 导入，在任务 metrics 中声明 category 和 requires。实验 YAML 的 metric_rules 可以扩展任务政策。文生音乐扩展标量通过 report_scalars 声明，如 `custom.score: {category: aesthetics, direction: higher}`；方向可为 higher/lower/descriptive。自定义 Metric 通过旧评分脚本在其独立环境执行，需安装 harness 基础依赖及对应插件。主指标 worker 不要求安装 harness 到评测环境。

复现实验需固定测试集、评分器版本、归一化/截取方式、候选数和随机种子。不同数据集和协议的结果不能直接与论文数字排名。人工听评由外部人员完成，不包含在本项目。
