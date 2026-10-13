# SongEval 本地接入

使用你已经跑通的仓库和 conda 环境，无需重新安装。默认所有评分器复用这一环境，按检查结果补充缺库；也可以将 evaluation_environment 改为 musicbench。当前任务固定包含 SongEval，不区分简单/复杂验证。完整手工准备与缺依赖排查见 [离线教程](METRIC_SETUP.md)。

在 configs/metrics.local.yaml 中配置：

```yaml
evaluation_environment: your_songeval_env
metric_options:
  songeval:
    repo: /data/musicbench-evaluators/SongEval
    muq_model: /data/musicbench-evaluators/MuQ/weights/muq
    device: auto
```

repo 包含官方 model.py、config.yaml、ckpt/model.safetensors。muq_model 是 MuQ-large-msd-iter 的完整本地目录（config.json 和权重），不填远端模型 ID。所有模型加载禁止联网；缺少文件会直接报错，不下载。模型可以保留在你对应库的目录中。

代码获取（仅联网准备机执行，已有仓库跳过）：

```bash
git clone https://github.com/ASLP-lab/SongEval.git /data/musicbench-evaluators/SongEval
git clone https://github.com/tencent-ailab/MuQ.git /data/musicbench-evaluators/MuQ
```

将已配好的代码、环境和本地权重复制到离线服务器，然后检查：

```bash
python scripts/setup_metrics.py --metrics songeval
./run.sh text2music
```

集成按官方方法使用 mono 24kHz 音频、MuQ hidden state 6、严格加载评分模型权重和 eval/inference 模式。结果保存五维分数、覆盖率、错误、checkpoint SHA256、上游提交和运行设备；不偷偷裁剪整曲。naturalness 只统计明确 has_vocal=true 的人声样本，纯音乐和未知标注显示不适用。

兼容旧的单指标 score.py 入口时，设置 MUSICBENCH_SONGEVAL_REPO、MUSICBENCH_SONGEVAL_ENV、MUSICBENCH_SONGEVAL_MUQ；后者同样为本地 MuQ 目录。新工作流优先使用上面的 YAML。

SongEval 预测美学感知，不能替代歌词、风格一致性评分或人工听评。上游许可证以你使用版本的仓库文件为准：[SongEval 官方仓库](https://github.com/ASLP-lab/SongEval)。
