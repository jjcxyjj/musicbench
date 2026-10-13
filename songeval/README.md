# songeval

代码参考：https://github.com/ASLP-lab/SongEval

本目录 score.py 是当前能力的直接评分函数；上游代码已放 upstream/，本地模型可放 weights/，也可在根目录 config.yaml 指定外部路径。

源码：

源码已放在本目录 upstream/；固定版本和来源见 SOURCE.json。服务器无需再克隆。

只在已激活的 songeval 环境运行根目录 main.py。模型与库手工配置，详细用法见 [根目录说明](../README.md)。

模型放置：评分头手工放在 upstream/ckpt/model.safetensors（创建 ckpt 目录）；MuQ 编码器的完整本地目录填根 config.yaml 的 songeval.muq_model。若 repo 指向已有 SongEval 仓库，评分头放其 ckpt/model.safetensors。本次仅附 config.yaml、model.py 等源码，没有评分头权重。依赖见 upstream/requirements.txt；已跑通的环境无需重装。
