#!/usr/bin/env python
"""
Figure 3 of section 2.x: dynamical regimes via window RQA.

Panels:
  A — cohort PCA of windowed RQA vectors, colored by window speed
       (one point per window across 16 NOF Day-1 mice)
  B — per-metric Pearson r (RQA ↔ speed), real vs shuffled,
       mean ± 95% CI, per-mouse points overlaid
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
RESULTS = HERE / 'results'
OUT = RESULTS / 'window_rqa_clustering'
DATA = OUT / 'data'
CSV = OUT / 'per_metric_speed.csv'

MEASURES = ['DET', 'LAM', 'ENTR', 'TT', 'L_mean', 'L_max']


def load_cohort_windows():
    """Stack RQA vectors and speeds across all per_window_*.npz."""
    Xs, ss = [], []
    sessions = sorted(DATA.glob('per_window_NOF_*.npz'))
    for p in sessions:
        d = np.load(p, allow_pickle=True)
        Xs.append(d['rqa_real'])
        ss.append(d['speed_real'])
    X = np.vstack(Xs); s = np.concatenate(ss)
    return X, s, len(sessions)


def main():
    # --- PCA cohort ---
    X, speed, n_sess = load_cohort_windows()
    print(f'Cohort: {n_sess} mice, {len(X)} windows total')
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit(Xs)
    pc = pca.transform(Xs)
    pc1_var = pca.explained_variance_ratio_[0]
    pc2_var = pca.explained_variance_ratio_[1]
    print(f'  PC1 = {pc1_var:.3f}, PC2 = {pc2_var:.3f} of variance')

    # --- Per-metric cohort: load CSV ---
    metrics = defaultdict(lambda: {'real': [], 'shuf': []})
    with open(CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row['metric']]['real'].append(float(row['r_real']))
            metrics[row['metric']]['shuf'].append(float(row['r_shuf']))
    for m in metrics:
        metrics[m]['real'] = np.array(metrics[m]['real'])
        metrics[m]['shuf'] = np.array(metrics[m]['shuf'])
    n_mice = len(next(iter(metrics.values()))['real'])

    # === Figure ===
    # n_bars + 1 column for colorbar
    n_bars = len(MEASURES)
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, n_bars + 1, height_ratios=[1.5, 1],
                          width_ratios=[1]*n_bars + [0.08],
                          wspace=0.55, hspace=0.55)

    # Panel A: PCA spans full bar width
    axA = fig.add_subplot(gs[0, :n_bars])
    cax = fig.add_subplot(gs[0, n_bars])
    sp_lo, sp_hi = np.percentile(speed, [2, 98])
    sc = axA.scatter(pc[:, 0], pc[:, 1], c=speed, cmap='plasma',
                     s=22, alpha=0.65, edgecolors='none',
                     vmin=sp_lo, vmax=sp_hi)
    axA.set_xlabel(f'PC1 ({100*pc1_var:.1f}%)', fontsize=11)
    axA.set_ylabel(f'PC2 ({100*pc2_var:.1f}%)', fontsize=11)
    axA.set_title(
        f'A. PCA векторов RQA-окон, цвет — скорость '
        f'({len(X)} окон, {n_sess} сессий)',
        fontsize=12, loc='left')
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label('скорость в окне (см/с)', fontsize=10)

    # Panel B: per-metric bars
    rng = np.random.default_rng(0)
    def _ci95(a):
        return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))

    for col, m in enumerate(MEASURES):
        ax = fig.add_subplot(gs[1, col])
        r = metrics[m]['real']
        s = metrics[m]['shuf']
        m_r, ci_r = _ci95(r)
        m_s, ci_s = _ci95(s)
        ax.bar([0], [m_r], width=0.6, yerr=[ci_r],
               color='#2CA02C', edgecolor='black', linewidth=0.5,
               capsize=4, label='Реальные')
        ax.bar([1], [m_s], width=0.6, yerr=[ci_s],
               color='#888888', edgecolor='black', linewidth=0.5,
               capsize=4, label='Перемеш.')
        jit_r = (rng.random(len(r)) - 0.5) * 0.22
        jit_s = (rng.random(len(s)) - 0.5) * 0.22
        ax.scatter(np.full(len(r), 0) + jit_r, r, s=18,
                   facecolor='white', edgecolor='black', linewidth=0.5,
                   alpha=0.85, zorder=3)
        ax.scatter(np.full(len(s), 1) + jit_s, s, s=18,
                   facecolor='white', edgecolor='black', linewidth=0.5,
                   alpha=0.85, zorder=3)
        ax.axhline(0, color='k', lw=0.6, alpha=0.5, ls='--')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Реал.', 'Перем.'], fontsize=9)
        ax.set_title(m, fontsize=11)
        if col == 0:
            ax.set_ylabel('Пирсон $r$ (со скоростью)', fontsize=10)
        # individual legends omitted; common figure-level legend appears below
        # значимость: Уилкоксон real vs 0 (метрики характеризуют регулярность,
        # которая отрицательно коррелирует со скоростью)
        p = stats.wilcoxon(r, alternative='less').pvalue
        if p < 1e-4: mark = '***'
        elif p < 1e-2: mark = '**'
        elif p < 0.05: mark = '*'
        else: mark = 'n.s.'
        all_vals = np.concatenate([r, s, [m_r + ci_r, m_s + ci_s]])
        ymax = all_vals.max(); ymin = min(all_vals.min(), 0)
        rng_y = ymax - ymin if ymax > ymin else 0.1
        ax.set_ylim(ymin - rng_y * 0.05, ymax + rng_y * 0.22)
        ax.plot([0, 1], [ymax + rng_y * 0.08]*2, color='black', lw=0.8)
        ax.text(0.5, ymax + rng_y * 0.11, mark, ha='center',
                fontsize=11, fontweight='bold')

    fig.text(0.5, 0.43,
             f'B. Корреляции RQA-метрик со скоростью движения '
             f'(n = {n_mice} сессий, 16 мышей × 4 дня, mean ± 95% ДИ)',
             ha='center', fontsize=12)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#2CA02C', edgecolor='black', label='Реальные данные'),
        Patch(facecolor='#888888', edgecolor='black', label='Перемешанные данные'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), ncol=2,
               fontsize=10, frameon=False)

    fig.savefig(OUT / 'fig_dynamics.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUT / "fig_dynamics.png"}')


if __name__ == '__main__':
    main()
