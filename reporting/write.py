"""Write a small CSV, Markdown report, and one chart per available score group."""
import csv
import math
import os
from pathlib import Path


def write_report(result, output):
    output = Path(output)
    lines = ['# MusicBench 既有音频评测', '', result['note'], '',
             '| 算法 | 指标 | 状态 | 分数 / 原因 |', '|---|---|---|---|']
    table = []
    for algorithm, metrics in result['algorithms'].items():
        for metric, entry in metrics.items():
            scores = entry.get('scores', {})
            display = ', '.join(f'{k}={v:.6f}' for k, v in scores.items() if isinstance(v, (float, int)) and math.isfinite(v))
            reason = entry.get('reason') or ''
            if entry['status'] == 'disabled':
                reason = '尚未启用；配好该能力后加入 config.yaml 的 enabled_metrics'
            if not display:
                display = reason or '没有有效分数'
            cell = lambda text: str(text).replace('|', '/').replace('\n', ' ')
            lines.append('| ' + ' | '.join(map(cell, [algorithm, metric, entry['status'], display])) + ' |')
            for key, value in scores.items():
                table.append([algorithm, metric, key, '' if value is None else value, entry['status']])
            for error in entry.get('errors', []):
                lines.append(f"\n错误：{cell(algorithm)} / {cell(metric)} / {cell(error)}\n")
            for sample in entry.get('per_sample', []):
                if sample.get('caption'):
                    lines.append(f"\n描述 {cell(sample['id'])}：{cell(sample['caption'])}\n")
            if entry.get('model'):
                lines.append(f"\n模型记录：{cell(entry['model'])}\n")
    with (output / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['algorithm', 'metric', 'score', 'value', 'status'])
        writer.writerows(table)
    if table:
        os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for metric in result['enabled_metrics']:
            keys = list(dict.fromkeys(r[2] for r in table if r[1] == metric))
            if not keys:
                continue
            fig, axes = plt.subplots(len(keys), 1, figsize=(9, 3 * len(keys)), squeeze=False)
            for ax, key in zip(axes[:, 0], keys):
                values = [r for r in table if r[1:3] == [metric, key] and isinstance(r[3], (int, float))]
                labels = [r[0] for r in values]
                ax.bar(range(len(values)), [r[3] for r in values])
                ax.set_xticks(range(len(values)), labels)
                ax.set_title(metric + '.' + key + (' (descriptive)' if key == 'PC' else ''))
            fig.tight_layout()
            filename = metric + '.jpg'
            fig.savefig(output / filename, dpi=150)
            plt.close(fig)
            lines += ['', f'![{metric}]({filename})']
    lines += ['', '注意：各指标分别解读，不计算综合总分。歌词错误率受识别器影响；captioner 描述不是匹配评分；FAD 是集合分布距离。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
