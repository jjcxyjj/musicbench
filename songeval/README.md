# songeval

代码参考：https://github.com/ASLP-lab/SongEval

本目录 score.py 是当前能力的直接评分函数；上游代码可放 upstream/，本地模型可放 weights/，也可在根目录 config.yaml 指定外部路径。

联网准备机获取代码：

```bash
git clone https://github.com/ASLP-lab/SongEval.git songeval/upstream
```

只在已激活的 songeval 环境运行根目录 main.py。模型与库手工配置，详细用法见 [根目录说明](../README.md)。
