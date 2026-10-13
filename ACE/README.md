# ACE：两个音频标注模型

这里的两个能力是 **acestep-captioner**（音频生成音乐描述）与 **acestep-transcriber**（音频识别歌词），不是 ACE 生成模型内部的 PMI / DiT attention 评分。

官方模型说明：
- https://huggingface.co/ACE-Step/acestep-captioner
- https://huggingface.co/ACE-Step/acestep-transcriber

两张模型卡均明确写明 Usage 与 Qwen2.5-Omni-7B 相同。未找到各自独立的官方 GitHub 推理仓库；通用推理代码在 https://github.com/QwenLM/Qwen2.5-Omni 。生成项目本身在 https://github.com/ace-step/ACE-Step-1.5 ，不要把生成用 DiT/5Hz-LM 权重填进这两个模型路径。

## 代码与权重放置

```text
ACE/
  inference.py                    # 两者共用的 Qwen2.5-Omni 本地加载
  score.py                        # main.py 调用入口
  acestep-captioner/run.py         # 独立运行音乐描述
  acestep-captioner/weights/       # 手工放完整 captioner 模型目录
  acestep-transcriber/run.py       # 独立运行歌词识别
  acestep-transcriber/weights/     # 手工放完整 transcriber 模型目录
  internal_pmi.py                 # 之前内部 PMI 实验代码，当前入口不调用
```

权重目录包含原模型的 config.json、分片 safetensors 与索引、tokenizer、processor/preprocessor 等文件。这里给出 Hugging Face 页面用于辨认官方模型，不要求服务器访问它；你从自己准备的本地位置复制完整模型即可。也可把 config.yaml 改成已有绝对路径。

参考源码（尚未克隆）：

参考源码尚未拉取；在项目根目录执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/QwenLM/Qwen2.5-Omni.git ACE/upstream
```

只克隆 GitHub 源码，不下载模型参数。

运行依赖：torch、支持 Qwen2_5OmniForConditionalGeneration/Qwen2_5OmniProcessor 的 transformers、accelerate、qwen-omni-utils 及音频解码依赖。按照 Qwen 仓库说明在兼容的联网 Linux 机器准备 wheelhouse，再复制到服务器当前 songeval 环境安装；不要盲目升级已跑通的依赖。缺少上述类时说明 transformers 版本不支持接口。首次运行不依赖 flash-attn，采用 SDPA。

```bash
conda activate songeval
python -m pip install --no-index --find-links /data/wheelhouse transformers accelerate qwen-omni-utils
python ACE/acestep-captioner/run.py --model /data/models/acestep-captioner --audio /data/generated/song.wav --output /data/results/caption.json
python ACE/acestep-transcriber/run.py --model /data/models/acestep-transcriber --audio /data/generated/song.wav --output /data/results/transcript.json
```

两个命令依次运行，不同时加载两个模型。只生成文本：disable_talker() 与 return_audio=False；全部 from_pretrained 均使用本地目录和 local_files_only=True，入口禁止网络访问。

官方提示词分别是 `*Task* Describe this audio in detail` 和 `*Task* Transcribe this audio in detail`，音频通过 processor 的 audio 字段提供，不用自行拼接 `<audio>` 字符串。

## 在主入口启用

在 config.yaml 填好 ace_captioner.model / ace_transcriber.model 后，加入 enabled_metrics，或：

```bash
python main.py --input /data/generated/results.json --metrics ace_captioner ace_transcriber
```

Captioner 生成描述并保存在 scores.json/report.md，**自身不是风格匹配分数**；风格主指标仍为 CLAP / MuQ-MuLan。Transcriber 仅看到最终音频，不接收目标歌词，输出 `# Lyrics` 部分再与目标歌词比较 PER/CER/WER。结构标签不算作要唱的字词。没有标准 Lyrics 标题、输出为空、疑似长度截断时记录失败及原文，不假装有效分数。

主入口歌词评分只选 has_vocal=true 且有目标歌词的记录；独立运行 transcriber 不要求目标歌词。中文使用 pypinyin，英文使用 phonemizer+espeak-ng；当前仅定义 zh/en 错误率协议。英文可设置 PHONEMIZER_ESPEAK_LIBRARY 指向本地动态库。这两个模型不需要 ACE 生成服务，不需要 DiT/VAE/5Hz LM。实际权重与单环境兼容性须在服务器验证；当前只有接口测试。
