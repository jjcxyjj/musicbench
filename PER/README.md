# PER：歌词音素错误率

PER 是计算规则，不是一个单独的预训练模型。当前独立能力采用 faster-whisper 识别最终混音，再与目标歌词比较，绝不把目标歌词传给识别器。

- ASR：https://github.com/SYSTRAN/faster-whisper
- 中文 G2P：https://github.com/mozillazg/python-pinyin
- 英文 G2P：https://github.com/bootphon/phonemizer
- 英文系统依赖：https://github.com/espeak-ng/espeak-ng

参考源码尚未拉取；在项目根目录执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/SYSTRAN/faster-whisper.git PER/upstream
```

只克隆 GitHub 源码，不下载模型参数。

手工准备 faster-whisper / CTranslate2 格式完整本地模型（含 model.bin、config.json、tokenizer 等），放 PER/weights/large-v3 或填 per.model。依赖 faster-whisper、pypinyin；英文还需 phonemizer 与 espeak-ng 动态库，可设置 per.espeak_library。服务器通过已准备的 wheelhouse 离线安装，不访问模型站。

中文 PER 使用严格声母+带声调韵母，同时报告 CER；英文使用无重音 espeak 音素，同时报告 WER。错误率为 (替换+删除+插入)/参考单位数，可大于 1，越低越好。按参考单位微平均，不能把英文 WER 与中文 CER 混成一个分数。G2P 失败保留识别文本和字词错误率，标明部分失败。

当前直接识别最终混音，没有自动分轨。伴奏和识别器错误会影响结果。这是“基于 ASR 的歌词匹配代理指标”，不能认定为听觉真值。ACE transcriber 也可使用本目录相同错误率规则；两个识别器结果分开报告。
