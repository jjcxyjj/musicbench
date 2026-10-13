# CLAP：音频与风格描述相似度

官方仓库：https://github.com/LAION-AI/CLAP

源码已放在本目录 upstream/；固定版本和来源见 SOURCE.json。服务器无需再克隆。

当前 score.py 使用 transformers 的 ClapModel / ClapProcessor，依赖 torch、transformers、librosa。完整本地 Transformers 格式权重、配置、tokenizer/processor 文件放 CLAP/weights/clap，或 config.yaml clap.model 指定已有目录；原始 LAION 单个 .pt 权重不能直接填入此适配器。

输入是最终音频 + 风格描述，不把歌词混入风格提示。配好后启用 clap。FAD 默认复用相同 CLAP 音频编码器；不自动下载模型。
