"""Category reports for primary music-generation metrics; no combined ranking."""
from pathlib import Path
import math
from .catalog import CATEGORIES


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', '<br>')


def render(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    algorithms, definitions = data['algorithms'], data['metric_definitions']
    from musicbench.core.runner import _flatten
    flat = {label: _flatten(scores) for label, scores in algorithms.items()}
    lines = ['# MusicBench 文生音乐自动评测报告', '',
             f"样本输出数：{data.get('sample_count')}；随机种子：{data.get('seed')}；每条输入生成次数：{data.get('repetitions', 1)}。", '',
             '所有算法共用测试清单；全部生成结果参与评测。各项单独报告，不计算综合总分。', '',
             '## 指标执行计划', '', '| 评测器 | 分类 | 适用样本数 | 状态 / 原因 |', '| --- | --- | --- | --- |']
    for name, plan in data.get('evaluation_plan', {}).items():
        lines.append(f"| {name} | {CATEGORIES.get(plan['category'], plan['category'])} | {len(plan['eligible_ids'])} | {cell(plan.get('reason') or plan['status'])} |")
    images = []
    for category, title in CATEGORIES.items():
        keys = [k for k, spec in definitions.items() if spec['category'] == category]
        lines.extend(['', f'## {title}', '', '| 指标 | 方向 | ' + ' | '.join(map(cell, algorithms)) + ' |',
                      '| --- | --- | ' + ' | '.join('---' for _ in algorithms) + ' |'])
        findings = []
        for key in keys:
            direction = definitions[key]['direction']
            values = []
            group = key.split('.')[0]
            for label, groups in algorithms.items():
                result = groups.get(group, {})
                value = flat[label].get(key)
                if result.get('status') == 'not_applicable':
                    display = '不适用'
                elif numeric(value):
                    display = f'{value:.6f}'
                    if key == 'songeval.naturalness':
                        display += f'（适用 n={result.get("naturalness_n_expected", "?")}）'
                    if result.get('available') is False or result.get('coverage', 1) < 1:
                        display += '（结果不完整）'
                elif key == 'songeval.naturalness' and result.get('available') and result.get('naturalness_n_expected') == 0:
                    display = '不适用（无人声标注）'
                elif key.startswith('lyrics.') and result.get('available') and value is None:
                    display = '不适用（语言）'
                else:
                    display = '未得到有效分数'
                values.append(display)
            lines.append('| ' + key + ' | ' + {'higher': '↑', 'lower': '↓', 'descriptive': '描述性'}[direction] + ' | ' + ' | '.join(values) + ' |')
            valid = {label: flat[label][key] for label in algorithms if numeric(flat[label].get(key))
                     and algorithms[label].get(group, {}).get('available') is not False
                     and algorithms[label].get(group, {}).get('coverage', 1) == 1}
            if len(valid) > 1 and direction != 'descriptive':
                best = (max if direction == 'higher' else min)(valid.values())
                winners = ', '.join(label for label, value in valid.items() if value == best)
                findings.append(f'\n观察到的 {key} 最优/并列：{cell(winners)}（{best:.6f}）；有效算法 {len(valid)}/{len(algorithms)}。\n')
        lines.extend(findings)
        drawable = [key for key in keys if any(numeric(row.get(key)) for row in flat.values())]
        filename = category + '.png'
        if drawable:
            plot(output / filename, flat, drawable, definitions, category)
            images.append(filename)
            lines.extend(['', f'![{title}]({filename})'])
        else:
            lines.extend(['', '该类别没有可绘制的有效分数。'])
    lines.extend(['', '## 可用性与错误', ''])
    issues = False
    for label, groups in algorithms.items():
        for name, result in groups.items():
            if result.get('status') == 'not_applicable':
                lines.append(f'- {cell(label)}/{name}：不适用，{cell(result.get("reason"))}。')
            elif result.get('available') is False or result.get('coverage', 1) < 1:
                issues = True
                lines.append(f'- **失败或不完整：{cell(label)}/{name}**；覆盖率 {result.get("coverage", 0):.1%}；'
                             f'{cell(result.get("reason") or result.get("errors"))}。')
    if not issues:
        lines.append('所有应运行指标均已完整返回。')
    lines.extend(['', '## 歌词识别与差异', '',
                  '以下为基于自动识别的歌词一致性，识别或人声分离错误也可能造成差异。PER 不表示时间同步误差。', ''])
    for label, groups in algorithms.items():
        lyric = groups.get('lyrics', {})
        for row in lyric.get('per_sample', []):
            lines.extend(['', f"### {cell(label)} / {cell(row['id'])}", '',
                          f"目标：{cell(row['reference'])}", '', f"识别：{cell(row['transcript'])}", '',
                          '| 操作 | 目标 | 识别 |', '| --- | --- | --- |'])
            for difference in row.get('lexical_alignment', {}).get('differences', []):
                lines.append(f"| {difference['operation']} | {cell(difference['reference'])} | {cell(difference['hypothesis'])} |")
            if row.get('per_error'):
                lines.append('\nPER 失败：' + cell(row['per_error']))
    lines.extend(['', '## 运行统计', '', '| 算法 | 推理耗时（秒） | 输出数 | 状态 |', '| --- | --- | --- | --- |'])
    for label, row in data.get('runtime', {}).items():
        lines.append(f"| {cell(label)} | {row.get('inference_seconds', '—')} | {row.get('generated_count', 0)} | {row['status']} |")
    lines.extend(['', '## 解释与复现边界', '',
                  '- Audiobox 和 SongEval 是自动感知预测分数，不能替代人工听评；PC 表示复杂程度，不预设越高越好。',
                  '- SongEval 的 Naturalness 针对人声呼吸和分句，仅统计 has_vocal=true 的样本。',
                  '- CLAP、MuQ-MuLan 读取 style_prompt，不读取 lyrics；整体相似度不能保证每个条件都满足。',
                  '- FAD 是集合分布距离，当前使用 CLAP audio features；不同编码器、样本量或参考集合的数字不可直接混比。',
                  '- PER 采用明确的 G2P 协议，未声称与 ACE-Lyric-Reward 或其他论文的 PER 完全等价。',
                  '- 评分器、协议和逐样本结果保存在 metric_results；参考路径保存在 reference_manifest.json；运行命令和错误见 logs。',
                  '- 不适用与执行失败分别记录；有效样本覆盖率不同的均值不应直接宣称优劣。',
                  '- 各指标观察到的差异不等于统计显著性；不同测试集的结果不可直接与论文表格中的数值排名。'])
    for filename in ('report.md', 'report.log'):
        (output / filename).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Keep a predictable single preview image as well as per-category images.
    all_keys = [k for k in definitions if any(numeric(row.get(k)) for row in flat.values())]
    plot(output / 'comparison.png', flat, all_keys, definitions, 'All primary metrics')


def plot(path, rows, keys, definitions, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = max(1, len(keys))
    cols = min(3, panels)
    fig, axes = plt.subplots((panels + cols - 1) // cols, cols,
                             figsize=(5 * cols, 3.6 * ((panels + cols - 1) // cols)), squeeze=False)
    axes = list(axes.flat)
    labels = list(rows)
    for ax, key in zip(axes, keys):
        values = [rows[label].get(key) for label in labels]
        ax.bar(labels, [v if numeric(v) else float('nan') for v in values])
        ax.set_title(key + ' (' + definitions[key]['direction'] + ')', fontsize=9)
        ax.tick_params(axis='x', rotation=25)
        ax.margins(y=0.15)
        for i, value in enumerate(values):
            if numeric(value):
                ax.annotate(f'{value:.3f}', (i, value), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8)
    if not keys:
        axes[0].text(0.5, 0.5, 'No valid scores; see report for failures / N/A', ha='center', va='center', transform=axes[0].transAxes)
    for ax in axes[max(1, len(keys)):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
