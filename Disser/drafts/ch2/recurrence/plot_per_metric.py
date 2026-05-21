#!/usr/bin/env python
"""Cleaner per-metric figure. Reads per_metric_speed.csv from per_window_analysis."""
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).parent
CSV = HERE / 'results' / 'window_rqa_clustering' / 'per_metric_speed.csv'
OUT = HERE / 'results' / 'window_rqa_clustering' / 'fig_per_metric_speed.png'

# Load
real = defaultdict(list)
shuf = defaultdict(list)
with open(CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        real[row['metric']].append(float(row['r_real']))
        shuf[row['metric']].append(float(row['r_shuf']))

# Order metrics by absolute median r in real
metrics = list(real.keys())
metrics.sort(key=lambda m: abs(np.median(real[m])), reverse=True)
n_mice = len(real[metrics[0]])

# Figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel A: per-mouse dot plot per metric (real vs shuffled)
ax = axes[0]
x = np.arange(len(metrics))
w = 0.18

for i, m in enumerate(metrics):
    # real - red dots, jittered
    r = np.array(real[m])
    jit = (np.random.RandomState(i).rand(len(r)) - 0.5) * 0.18
    ax.scatter(np.full(len(r), i - w) + jit, r, s=22, c='#cc3344',
               edgecolor='black', linewidth=0.4, alpha=0.85, zorder=3)
    # shuf - gray dots
    s = np.array(shuf[m])
    jit = (np.random.RandomState(i + 100).rand(len(s)) - 0.5) * 0.18
    ax.scatter(np.full(len(s), i + w) + jit, s, s=22, c='#bbbbbb',
               edgecolor='black', linewidth=0.4, alpha=0.85, zorder=3)
    # median bars
    ax.plot([i - w - 0.13, i - w + 0.13], [np.median(r)]*2,
            color='black', lw=2, zorder=4)
    ax.plot([i + w - 0.13, i + w + 0.13], [np.median(s)]*2,
            color='black', lw=2, zorder=4)

ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Pearson r (RQA metric ↔ window speed)')
ax.set_title(f'A. Per-mouse correlation per metric (n={n_mice} mice)\n'
             f'Red = real, gray = shuffled; bars = median', fontsize=11, loc='left')
ax.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#cc3344',
               markersize=9, markeredgecolor='k', label='Real'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#bbbbbb',
               markersize=9, markeredgecolor='k', label='Shuffled'),
], loc='upper right', fontsize=10)

# Panel B: median r ± IQR
ax = axes[1]
med_r = [np.median(real[m]) for m in metrics]
med_s = [np.median(shuf[m]) for m in metrics]
q1_r  = [np.percentile(real[m], 25) for m in metrics]
q3_r  = [np.percentile(real[m], 75) for m in metrics]
q1_s  = [np.percentile(shuf[m], 25) for m in metrics]
q3_s  = [np.percentile(shuf[m], 75) for m in metrics]

err_r = [np.array(med_r) - np.array(q1_r), np.array(q3_r) - np.array(med_r)]
err_s = [np.array(med_s) - np.array(q1_s), np.array(q3_s) - np.array(med_s)]

ax.bar(x - 0.2, med_r, 0.38, yerr=err_r, color='#cc3344',
       edgecolor='black', linewidth=0.5, capsize=4, label='Real')
ax.bar(x + 0.2, med_s, 0.38, yerr=err_s, color='#bbbbbb',
       edgecolor='black', linewidth=0.5, capsize=4, label='Shuffled')
ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.7)

# Significance markers per metric (Wilcoxon real vs 0)
for i, m in enumerate(metrics):
    r = np.array(real[m])
    # one-sided away-from-zero
    if np.median(r) < 0:
        p = stats.wilcoxon(r, alternative='less').pvalue
    else:
        p = stats.wilcoxon(r, alternative='greater').pvalue
    if p < 1e-3:
        mark = '***'
    elif p < 1e-2:
        mark = '**'
    elif p < 0.05:
        mark = '*'
    else:
        mark = 'n.s.'
    y_pos = (med_r[i] - 0.06) if med_r[i] < 0 else (med_r[i] + 0.04)
    ax.text(i - 0.2, y_pos, mark, ha='center', fontsize=10, fontweight='bold',
            color='black')

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Median Pearson r ± IQR')
ax.set_title('B. Cohort median per metric (Wilcoxon signed-rank vs 0)',
             fontsize=11, loc='left')
ax.legend(loc='best', fontsize=10)

fig.suptitle('Per-metric speed correlation across NOF Day-1 cohort '
             f'({n_mice} mice)', fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches='tight')
print(f'Saved: {OUT}')
