"""Export project workflows as high-resolution JPG files."""
from pathlib import Path
import argparse
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def font_path():
    for path in ('/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/STHeiti Light.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
        if Path(path).is_file():
            return path
    for path in font_manager.findSystemFonts():
        if any(name in path.lower() for name in ('notosanscjk', 'sourcehansans', 'wqy', 'simhei')):
            return path
    raise RuntimeError('Install Noto Sans CJK or supply --font')


INK, MUTED, BLUE, GREEN, BORDER = '#182b49', '#5b6b81', '#315da8', '#197766', '#cbd5e1'


def canvas(title, subtitle, font):
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.text(.5, .965, title, ha='center', va='top', fontsize=25, color=INK, fontproperties=font)
    ax.text(.5, .91, subtitle, ha='center', va='center', fontsize=13, color=MUTED, fontproperties=font)
    fig.subplots_adjust(left=.025, right=.975, top=.98, bottom=.025)
    return fig, ax


def box(ax, x, y, w, h, title, lines, font, color=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012', facecolor='#f5f8fc', edgecolor=BORDER, linewidth=1.1))
    ax.text(x + .02, y + h - .023, title, fontsize=16, color=color, va='top', fontproperties=font)
    ax.text(x + .02, y + h - .062, '\n'.join(lines), fontsize=12.5, color=INK, va='top', linespacing=1.5, fontproperties=font)


def arrow(ax, points):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, linewidth=1.5)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>', mutation_scale=16, color=MUTED, linewidth=1.5))


def draw(destination, title, subtitle, items, font):
    fig, ax = canvas(title, subtitle, font)
    ys = [.73, .56, .39, .22, .05]
    for i, ((heading, lines), y) in enumerate(zip(items, ys)):
        box(ax, .07, y, .86, .125, heading, lines, font)
        if i < len(items) - 1:
            arrow(ax, [(.5, y - .008), (.5, ys[i + 1] + .133)])
    fig.savefig(destination, dpi=180, facecolor='white', pil_kwargs={'quality': 95})
    plt.close(fig)


def main():
    font = font_manager.FontProperties(fname=font_path())
    draw(ROOT / 'workflow_overview.jpg', 'MusicBench 简化流程',
         '一个 songeval 环境 · 只评估已有音频 · main.py 直接顺序调用', [
        ('1 读取算法输出 JSON', ['audio_path 必须指向实际生成音频；algorithm 只作为分组标签', '读取可选歌词、风格描述、GT 音频与原始标注']),
        ('2 交互确认三项输入', ['有没有目标歌词？有没有 GT / 真实参考音频？有没有风格描述？', '纯音乐不要求歌词；没有对应输入的指标记为不适用']),
        ('3 根据输入和已启用能力选择指标', ['config.yaml 默认启用 SongEval、Audiobox，其他能力配好后逐项加', '已附 SongEval / CLAP / Audiobox / MuQ 源码；其余见 README 克隆指令']),
        ('4 在当前环境依次评分', ['main.py → 对应 score.py；ACE 使用 Qwen2.5-Omni；不切换环境', '一项结束清理对象与 CUDA 缓存，再运行下一项']),
        ('5 保存分项结果', ['JSON、CSV、JPG、Markdown 报告；错误与不适用分别记录', '不计算综合质量总分；旧编排代码归档在 legacy/'])], font)
    draw(ROOT / 'text2music_evaluation.jpg', '已有音乐输出的自动评测',
         '输入决定适用范围 · 评分能力逐个跑通再启用', [
        ('1 美学评估：所有生成音频', ['SongEval 五维 + Audiobox CE / CU / PC / PQ', '人声自然度仅 has_vocal=true；PC 是描述性复杂度']),
        ('2 风格一致性：有风格描述', ['CLAP / MuQ-MuLan：最终音频 + style_prompt', '歌词不混入风格文本']),
        ('3 分布距离：有 GT 音频集合', ['FAD-CLAP（默认）：生成集合 vs 固定真实参考集合', '至少各两条；可选 MuQ，编码器与参考集不同不能混比']),
        ('4 歌词匹配：有人声且有目标歌词', ['PER（Whisper）或 ACE transcriber：音频识别 → PER / CER / WER', '识别时不提供目标歌词；纯音乐跳过；不同识别器分别报告']),
        ('5 音乐描述与分项报告', ['ACE captioner 输出音乐描述，不直接当作匹配分数', '部分源码已附，其余按 README 克隆；权重手工配置，服务器验证'])], font)
    print('Updated both workflow JPG files.')


if __name__ == '__main__':
    main()
