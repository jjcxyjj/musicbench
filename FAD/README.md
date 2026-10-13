# FAD：真实与生成音乐集合的分布距离

参考仓库：https://github.com/microsoft/fadtk

```bash
git clone https://github.com/microsoft/fadtk.git FAD/upstream
```

当前 score.py 自行计算 Fréchet 均值/协方差距离，不导入 fadtk，因此克隆仅供查阅。默认用 CLAP/weights/clap 的本地 Transformers 格式音频编码器，fad.model 指定实际路径；可设 embedding: muq_mulan 并填 model/audio_model/text_model 使用 MuQ-MuLan。无需独立 FAD 权重。

依赖 numpy、librosa、torch 及对应编码器库。每个算法至少两条生成音频，参考集合至少两条；这只是数学最低要求，不能保证小样本统计可靠。固定参考集合、编码器权重与预处理才可比较。当前每段音频得到一个特征，并非所有论文的分帧 FAD 实现，不能与其数值直接对齐。
