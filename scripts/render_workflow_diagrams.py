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


def overview(destination, font):
    fig, ax = canvas('MusicBench 总体流程', '任务规则驱动 · YAML 配置 · 同批随机数据 · 所有算法和评测器串行运行', font)
    left, right, width, height = .045, .55, .405, .12
    ys = [.74, .57, .40, .23]
    blocks = [
        ('1 选择任务与可用输入', ['读取任务的固定指标规则', '确认文本、歌词、参考音频或真实标签']),
        ('2 选择 / 登记算法', ['已有算法自动发现；新算法登记脚本或创建模板', '登记环境、工作目录和参数定义']),
        ('3 配置算法与测试参数', ['环境、显卡、模型权重、生成参数', '数据路径、样本数、随机种子、生成次数']),
        ('4 保存实验 YAML', ['校验配置；保存全部选择供复跑', '任务、算法、指标通过声明扩展']),
        ('5 准备共享测试清单', ['随机抽样一次，所有算法共用同批输入', '固定参考集合；保存清单与重复生成种子']),
        ('6 执行当前算法推理', ['在指定 conda 环境 / 显卡上生成', '校验输出 ID 与音频；进程结束释放显存']),
        ('7 串行运行任务评测器', ['按输入条件启用；每项使用指定环境 / 显卡', '保存分数、中间结果、版本和日志']),
        ('8 保存结果，检查下一算法', ['缺少输入：不适用；执行错误：记录失败', '当前算法评测结束后，才运行下一算法']),
    ]
    for col, x in enumerate((left, right)):
        for i, y in enumerate(ys):
            title, lines = blocks[col * 4 + i]
            box(ax, x, y, width, height, title, lines, font, BLUE if col == 0 else GREEN)
        for first, second in zip(ys, ys[1:]):
            arrow(ax, [(x + width / 2, first - .009), (x + width / 2, second + height + .009)])
    arrow(ax, [(left + width + .008, ys[-1] + height / 2), (.5, ys[-1] + height / 2), (.5, ys[0] + height / 2), (right - .008, ys[0] + height / 2)])
    arrow(ax, [(right + width + .008, ys[-1] + height / 2), (.985, ys[-1] + height / 2), (.985, ys[1] + height / 2), (right + width + .008, ys[1] + height / 2)])
    ax.text(.973, .47, '还有算法：继续', ha='center', va='center', rotation=90, fontsize=11, color=MUTED, fontproperties=font)
    box(ax, .15, .045, .70, .105, '9 全部算法结束：汇总图片与分析报告', ['分类展示指标、覆盖率和失败项；保留配置、原始结果与运行状态'], font, GREEN)
    arrow(ax, [(right + width / 2, ys[-1] - .009), (right + width / 2, .185), (.5, .185), (.5, .158)])
    ax.text(.60, .19, '无下一算法', fontsize=10.5, color=MUTED, fontproperties=font)
    fig.savefig(destination, dpi=180, facecolor='white', pil_kwargs={'quality': 95})
    plt.close(fig)


def text2music(destination, font):
    fig, ax = canvas('文生音乐自动评测流程', '输入分开保存：生成音频 / style_prompt / lyrics + language / 真实音乐参考集合', font)
    nodes = [
        (.725, '1 美学评估（含结构）｜固定运行', ['Audiobox：CE 欣赏程度 · CU 素材可用性 · PC 制作复杂度 · PQ 制作质量', 'SongEval：连贯性 · 音乐性 · 记忆性 · 结构清晰度 · 人声自然度', '人声自然度只统计明确有人声的样本；PC 为描述性指标']),
        (.505, '2 歌词匹配度｜有目标歌词时运行', ['统一音频处理协议 → 人声分离（或统一用混音）→ 音频独立转写', '目标歌词与转写规范化、对齐 → PER + 中文 CER / 英文 WER', '保存替换 / 删除 / 插入明细；目标歌词不作为识别提示']),
        (.285, '3 风格匹配度｜有风格描述时运行', ['style_prompt + 生成音乐 → CLAP 与 MuQ-MuLan 相似度', '歌词不混入风格文本；分别保留两个评分器的原始分数']),
        (.065, '4 与真实音乐分布的距离｜有真实参考集合时运行', ['当前算法的全部生成音频集合 + 固定真实音乐参考集合 → FAD', '使用 CLAP 音频特征；记录编码器、样本数与参考集合；不要求逐首配对']),
    ]
    for y, title, lines in nodes:
        box(ax, .065, y, .87, .155, title, lines, font, GREEN)
    for first, second in zip(nodes, nodes[1:]):
        arrow(ax, [(.5, first[0] - .009), (.5, second[0] + .164)])
    ax.text(.5, .025, '各阶段串行执行 · 无对应输入记为不适用 · 应运行却失败会保留错误 · 分项报告，不计算综合总分', ha='center', va='center', fontsize=12, color=MUTED, fontproperties=font)
    fig.savefig(destination, dpi=180, facecolor='white', pil_kwargs={'quality': 95})
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default=str(ROOT))
    p.add_argument('--font')
    args = p.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    font = font_manager.FontProperties(fname=args.font or font_path())
    overview(output / 'workflow_overview.jpg', font)
    text2music(output / 'text2music_evaluation.jpg', font)
    print(output / 'workflow_overview.jpg')
    print(output / 'text2music_evaluation.jpg')


if __name__ == '__main__':
    main()
