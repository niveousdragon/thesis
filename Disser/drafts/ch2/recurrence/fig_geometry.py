#!/usr/bin/env python
"""
Figure 2 of section 2.x: spatial structure recovery.

Panels:
  A — arena trajectory colored by 2D colorstamp (color legend for layout)
  B — population recurrence graph layout, nodes colored by same colorstamp
  C — cohort KNN-decoder error per mouse, real vs shuffled
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
import scipy.sparse as sp
import networkx as nx
from scipy import stats

DRIADA = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA / 'src'))
sys.path.insert(0, str(DRIADA / 'tools'))
sys.path.insert(0, str(DRIADA / 'science' / 'nof_place_reconstruction'))
from load_synchronized_experiments import load_experiment_from_npz
from plot_latent_space import position_to_color
from driada.dim_reduction.manifold_metrics import procrustes_analysis

DEMO_SESSION = 'NOF_H39_1D'   # best cohort session
DS = 5
PERCENTILE = 95.0
NOF = DRIADA / 'DRIADA data/NOF/SynchronizedData26_v1'
HERE = Path(__file__).parent
RESULTS = HERE / 'results'
OUT = RESULTS / 'window_rqa_clustering'
# Prefer all-days CSV if available; fall back to 1D
CSV_PATH = (OUT / 'geometry_cohort_pct95_all.csv'
            if (OUT / 'geometry_cohort_pct95_all.csv').exists()
            else OUT / 'geometry_cohort_pct95_1D.csv')


def build_layout(jrp_sparse, seed=42):
    """Sweep-optimised FA2 parameters (sR=1.0, gv=0.5)."""
    from fa2_modified import ForceAtlas2
    G = nx.from_scipy_sparse_array(jrp_sparse)
    fa2 = ForceAtlas2(outboundAttractionDistribution=True,
                      barnesHutOptimize=True, barnesHutTheta=1.2,
                      scalingRatio=1.0, gravity=0.5,
                      strongGravityMode=False, verbose=False)
    pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
    layout = np.array([pos[i] for i in range(G.number_of_nodes())])
    return layout


def main():
    # === Demo session: real layout colored by 2D colorstamp ===
    cache = RESULTS / DEMO_SESSION / f'mean_matrix_ds{DS}_k50_exp_md3.npz'
    cached = np.load(cache, allow_pickle=True)
    mm = cached['mean_matrix']
    taus = cached['taus']
    median_tau = int(np.median(taus))
    n = mm.shape[0]
    diag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < median_tau * 3
    m = mm.copy(); m[diag] = 0
    thr = np.percentile(m[m > 0], PERCENTILE)
    jrp = (m >= thr).astype(float)
    layout_raw = build_layout(sp.csr_matrix(jrp))

    # Trajectory aligned to mean-matrix indices
    exp = load_experiment_from_npz(NOF / f'{DEMO_SESSION}_aligned.npz', verbose=False)
    n_full = exp.calcium.data.shape[1] // DS + (
        1 if exp.calcium.data.shape[1] % DS else 0)
    offset = n_full - n
    x = exp.dynamic_features['x'].data[::DS][offset:offset + n]
    y = exp.dynamic_features['y'].data[::DS][offset:offset + n]
    xy = np.column_stack([x, y])
    colors = position_to_color(xy)

    # Procrustes-align FA2 layout to arena (rotation/reflection/scaling).
    # driada returns Y_aligned in arena coordinates (cm).
    # disparity = sqrt(sum residuals^2) → per-point RMSE = disparity/sqrt(n).
    layout, disparity, _ = procrustes_analysis(
        xy, layout_raw, scaling=True, reflection=True)
    rmse_cm = disparity / np.sqrt(len(xy))
    print(f'Demo session: Procrustes RMSE = {rmse_cm:.2f} cm '
          f'(arena side = 44 cm)')

    # === Cohort metrics: 4 columns (mantel_p, mantel_s, procr, knn_err) ===
    metrics = {'mantel_p': ([], []), 'mantel_s': ([], []),
               'procr': ([], []), 'knn_err': ([], [])}
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in metrics:
                metrics[k][0].append(float(row[f'r_{k}']))
                metrics[k][1].append(float(row[f's_{k}']))
    for k in metrics:
        metrics[k] = (np.array(metrics[k][0]), np.array(metrics[k][1]))
    n_sess = len(metrics['mantel_p'][0])

    # === Figure: top row arena + layout (large), bottom row 4 bar panels ===
    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1],
                          height_ratios=[1.5, 1], wspace=0.45, hspace=0.65)

    # Panel A: arena, large (two columns)
    axA = fig.add_subplot(gs[0, :2])
    axA.scatter(x, y, c=colors, s=8, alpha=0.85, edgecolors='none')
    axA.set_aspect('equal', adjustable='box')
    axA.set_xlabel('x (см)'); axA.set_ylabel('y (см)')
    axA.set_title('A. Траектория в арене',
                  fontsize=12, loc='left')

    # Panel B: layout after Procrustes alignment to arena
    axB = fig.add_subplot(gs[0, 2:])
    axB.scatter(layout[:, 0], layout[:, 1], c=colors, s=8, alpha=0.85,
                edgecolors='none')
    axB.set_aspect('equal', adjustable='box')
    axB.set_xlabel('x (см)'); axB.set_ylabel('y (см)')
    # match axis limits with panel A so panels look identical in scale
    xlim = axA.get_xlim(); ylim = axA.get_ylim()
    axB.set_xlim(xlim); axB.set_ylim(ylim)
    axB.set_title(
        f'B. Layout мультиплексного графа, RMSE = {rmse_cm:.1f} см',
        fontsize=12, loc='left')

    # Bottom row: 4 bar panels (mean ± 95% CI, paired Wilcoxon)
    def _ci95(a):
        m = a.mean()
        sem = a.std(ddof=1) / np.sqrt(len(a))
        return m, 1.96 * sem

    metric_titles = [
        ('mantel_p',   'Мандель (Пирсон)', 'greater', 'higher better'),
        ('mantel_s',   'Мандель (Спирмен)', 'greater', 'higher better'),
        ('procr',      'Procrustes',       'less',    'lower better'),
        ('knn_err',    'Ошибка $k$-NN (см)', 'less',  'lower better'),
    ]
    rng = np.random.default_rng(0)
    for col, (k, name, alt, _) in enumerate(metric_titles):
        ax = fig.add_subplot(gs[1, col])
        r, s = metrics[k]
        m_r, ci_r = _ci95(r)
        m_s, ci_s = _ci95(s)
        ax.bar([0], [m_r], width=0.6, yerr=[ci_r],
               color='#2CA02C', edgecolor='black', linewidth=0.5,
               capsize=4, label='Реальные', zorder=1)
        ax.bar([1], [m_s], width=0.6, yerr=[ci_s],
               color='#888888', edgecolor='black', linewidth=0.5,
               capsize=4, label='Перемеш.', zorder=1)
        # per-mouse points overlaid
        jit_r = (rng.random(len(r)) - 0.5) * 0.22
        jit_s = (rng.random(len(s)) - 0.5) * 0.22
        ax.scatter(np.full(len(r), 0) + jit_r, r, s=18,
                   facecolor='white', edgecolor='black', linewidth=0.5,
                   alpha=0.85, zorder=3)
        ax.scatter(np.full(len(s), 1) + jit_s, s, s=18,
                   facecolor='white', edgecolor='black', linewidth=0.5,
                   alpha=0.85, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Реал.', 'Перем.'], fontsize=10)
        ax.set_title(name, fontsize=11)
        p = stats.wilcoxon(r - s, alternative=alt).pvalue
        if p < 1e-4: mark = '***'
        elif p < 1e-2: mark = '**'
        elif p < 0.05: mark = '*'
        else: mark = 'n.s.'
        # y-limits using all points + CIs
        all_vals = np.concatenate([r, s, [m_r + ci_r, m_s + ci_s]])
        ymax = all_vals.max()
        ymin = min(all_vals.min(), 0)
        rng_y = ymax - ymin
        ax.set_ylim(ymin - rng_y * 0.05, ymax + rng_y * 0.22)
        ax.plot([0, 1], [ymax + rng_y * 0.08]*2, color='black', lw=0.8)
        ax.text(0.5, ymax + rng_y * 0.11, mark, ha='center',
                fontsize=11, fontweight='bold')
        # individual legends omitted; common figure-level legend appears below

    # group title between rows
    fig.text(0.5, 0.43,
             f'C. Групповые показатели восстановления геометрии '
             f'(n = {n_sess} сессий, 16 мышей × 4 дня, mean ± 95% ДИ; точки — сессии)',
             ha='center', fontsize=12)

    # common figure-level legend below the bar row
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#2CA02C', edgecolor='black', label='Реальные данные'),
        Patch(facecolor='#888888', edgecolor='black', label='Перемешанные данные'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), ncol=2,
               fontsize=10, frameon=False)

    fig.savefig(OUT / 'fig_geometry.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUT / "fig_geometry.png"}')


if __name__ == '__main__':
    main()
