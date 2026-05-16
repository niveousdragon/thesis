#!/usr/bin/env python
"""
Population recurrence network figure: REAL vs SHUFFLED side by side.

Layout (3 rows x 2 cols):
  Row 1: [Real network (position color)]  [Shuffled network (position color)]
  Row 2: [Real network (speed color)]     [Shuffled network (speed color)]
  Row 3: [Trajectory (position color)]    [Metrics bar chart (real vs shuffled)]

Usage:
    python plot_nof_recurrence_fig.py
    python plot_nof_recurrence_fig.py --session NOF_H01_1D --k 50
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import networkx as nx

DRIADA_ROOT = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA_ROOT / 'src'))
sys.path.insert(0, str(DRIADA_ROOT / 'tools'))
sys.path.insert(0, str(DRIADA_ROOT / 'science' / 'nof_place_reconstruction'))

from load_synchronized_experiments import load_experiment_from_npz
from plot_latent_space import position_to_color
from driada.recurrence.rqa import compute_rqa
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

OUT_BASE = Path(__file__).parent / 'results'
NOF_DATA_DIR = DRIADA_ROOT / 'DRIADA data' / 'NOF' / 'SynchronizedData26_v1'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--session', default='NOF_H32_1D')
    p.add_argument('--ds', type=int, default=5)
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--jrp-threshold', type=float, default=0.02)
    p.add_argument('--smooth-sigma', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def build_graph_and_layout(mm, med_tau, threshold, seed=42):
    """Binarize mean matrix, build graph, compute ForceAtlas2 layout."""
    n = mm.shape[0]
    diag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < med_tau * 3
    md = mm.copy()
    md[diag] = 0

    jrp = (md >= threshold).astype(float)
    jrp_sparse = sp.csr_matrix(jrp)
    G = nx.from_scipy_sparse_array(jrp_sparse)

    from fa2_modified import ForceAtlas2
    fa2 = ForceAtlas2(outboundAttractionDistribution=True,
                      barnesHutOptimize=True, barnesHutTheta=1.2,
                      scalingRatio=2.0, gravity=1.0, verbose=False)
    pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=200)

    return G, pos, jrp_sparse, md


def compute_metrics(G, pos_fa2, jrp_sparse, x_pos, y_pos, min_n, seed=42):
    """Compute spatial encoding metrics."""
    rng = np.random.default_rng(seed)

    # Position R^2
    fa2_coords = np.array([pos_fa2[i] for i in range(min_n)])
    pos_ref = np.column_stack([x_pos, y_pos])
    r2 = LinearRegression().fit(fa2_coords, pos_ref).score(fa2_coords, pos_ref)

    # Distance correlation
    n_sub = min(2000, min_n)
    idx = rng.choice(min_n, n_sub, replace=False)
    d_fa2 = pdist(fa2_coords[idx])
    d_pos = pdist(pos_ref[idx])
    rho_dist, _ = spearmanr(d_fa2, d_pos)

    # Neighbor dispersion
    jrp_csr = jrp_sparse.tocsr()
    dispersions = []
    for i in range(min_n):
        nb = jrp_csr[i].indices
        if len(nb) < 2:
            continue
        dx = x_pos[nb] - x_pos[nb].mean()
        dy = y_pos[nb] - y_pos[nb].mean()
        dispersions.append(np.sqrt((dx**2 + dy**2).mean()))
    mean_disp = np.mean(dispersions) if dispersions else np.nan

    # Null
    null_disps = []
    for _ in range(200):
        perm = rng.permutation(min_n)
        x_s, y_s = x_pos[perm], y_pos[perm]
        ds = []
        for i in range(min_n):
            nb = jrp_csr[i].indices
            if len(nb) < 2:
                continue
            dx = x_s[nb] - x_s[nb].mean()
            dy = y_s[nb] - y_s[nb].mean()
            ds.append(np.sqrt((dx**2 + dy**2).mean()))
        if ds:
            null_disps.append(np.mean(ds))
    null_disps = np.array(null_disps)
    disp_ratio = null_disps.mean() / mean_disp if mean_disp > 0 else 0

    return {
        'R2': r2,
        'dist_corr': rho_dist,
        'disp': mean_disp,
        'disp_null': null_disps.mean(),
        'disp_ratio': disp_ratio,
    }


def draw_network(ax, G, pos, node_list, xy, sizes, colors, title,
                 cmap=None, vmin=None, vmax=None, cbar_label=None):
    """Draw network panel."""
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.008, width=0.2,
                           edge_color='gray')
    if cmap is not None:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap=cmap, s=sizes,
                        vmin=vmin, vmax=vmax, edgecolors='none', alpha=0.85,
                        zorder=2)
        if cbar_label:
            plt.colorbar(sc, ax=ax, shrink=0.4, label=cbar_label, pad=0.01)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=sizes,
                   edgecolors='none', alpha=0.85, zorder=2)
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def add_colorstamp(ax):
    """Add 2D position colormap inset."""
    ng = 80
    xg = np.linspace(0, 1, ng)
    xxg, yyg = np.meshgrid(xg, xg)
    pg = np.column_stack([xxg.ravel(), yyg.ravel()])
    cg = np.zeros((ng * ng, 3))
    cg[:, 0] = pg[:, 0] * (1 - pg[:, 1]) + (1 - pg[:, 0]) * pg[:, 1]
    cg[:, 1] = pg[:, 0] * pg[:, 1] + (1 - pg[:, 0]) * (1 - pg[:, 1])
    cg[:, 2] = pg[:, 1]
    ax_s = inset_axes(ax, width="10%", height="10%", loc="upper right",
                      borderpad=0.3)
    ax_s.imshow(cg.reshape(ng, ng, 3), origin='lower', extent=[0, 1, 0, 1])
    ax_s.set_xticks([0, 1])
    ax_s.set_yticks([0, 1])
    ax_s.set_xticklabels(['L', 'R'], fontsize=6)
    ax_s.set_yticklabels(['B', 'T'], fontsize=6)
    ax_s.tick_params(length=0)


def main():
    args = parse_args()
    t0 = time.time()
    OUT = OUT_BASE / args.session
    OUT.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 1. Load data
    # ==================================================================
    npz_path = NOF_DATA_DIR / f'{args.session}_aligned.npz'
    exp = load_experiment_from_npz(npz_path, verbose=False)
    n_neurons = exp.calcium.data.shape[0]
    print(f'{args.session}: {n_neurons} neurons')

    calcium = exp.calcium.data[:, ::args.ds]
    if args.smooth_sigma > 0:
        calcium = np.array([gaussian_filter1d(c, args.smooth_sigma)
                            for c in calcium])
    n_frames_ds = calcium.shape[1]

    # Load caches
    cache_tag = f'ds{args.ds}_k{args.k}_exp_md3'
    real_cache = OUT / f'mean_matrix_{cache_tag}.npz'
    shuf_cache = OUT / f'mean_matrix_shuffled_{cache_tag}.npz'

    if not real_cache.exists() or not shuf_cache.exists():
        print(f'ERROR: run run_nof_recurrence.py first with matching params')
        return

    real_data = np.load(real_cache, allow_pickle=True)
    shuf_data = np.load(shuf_cache, allow_pickle=True)
    mm_real = real_data['mean_matrix']
    mm_shuf = shuf_data['mean_matrix']
    taus = real_data['taus']
    med_tau = int(np.median(taus))

    # Use smaller of the two sizes
    min_n = min(mm_real.shape[0], mm_shuf.shape[0])
    mm_real = mm_real[:min_n, :min_n]
    mm_shuf = mm_shuf[:min_n, :min_n]
    offset = n_frames_ds - min_n

    # Align behavioral variables
    def align(key):
        d = exp.dynamic_features[key].data
        a = d[::args.ds][offset:offset + min_n]
        if len(a) < min_n:
            a = np.pad(a, (0, min_n - len(a)), mode='edge')
        return a

    x_pos = align('x')
    y_pos = align('y')
    speed = align('speed')
    positions = np.column_stack([x_pos, y_pos])
    pos_colors = position_to_color(positions)

    # ==================================================================
    # 2. Build graphs + layouts
    # ==================================================================
    threshold = args.jrp_threshold

    print('Building real graph...')
    G_real, pos_real, jrp_real, md_real = build_graph_and_layout(
        mm_real, med_tau, threshold)
    print(f'  Real: {G_real.number_of_edges()} edges')

    print('Building shuffled graph...')
    G_shuf, pos_shuf, jrp_shuf, md_shuf = build_graph_and_layout(
        mm_shuf, med_tau, threshold)
    print(f'  Shuffled: {G_shuf.number_of_edges()} edges')

    # ==================================================================
    # 3. Metrics
    # ==================================================================
    print('\nMetrics (real):')
    m_real = compute_metrics(G_real, pos_real, jrp_real, x_pos, y_pos,
                             min_n, args.seed)
    for k, v in m_real.items():
        print(f'  {k}: {v:.3f}')

    print('Metrics (shuffled):')
    m_shuf = compute_metrics(G_shuf, pos_shuf, jrp_shuf, x_pos, y_pos,
                             min_n, args.seed + 1)
    for k, v in m_shuf.items():
        print(f'  {k}: {v:.3f}')

    # ==================================================================
    # 4. Figure
    # ==================================================================
    fig, axes = plt.subplots(3, 2, figsize=(20, 28))

    # Helper
    def get_node_data(G, pos):
        nl = list(G.nodes())
        xy = np.array([pos[nd] for nd in nl])
        deg = dict(G.degree())
        mx = max(deg.values()) if deg else 1
        sz = np.array([10 + 30 * deg.get(nd, 0) / mx for nd in nl])
        return nl, xy, sz

    nl_r, xy_r, sz_r = get_node_data(G_real, pos_real)
    nl_s, xy_s, sz_s = get_node_data(G_shuf, pos_shuf)

    # Row 1: position color
    draw_network(axes[0, 0], G_real, pos_real, nl_r, xy_r, sz_r,
                 pos_colors[nl_r],
                 f'Real (n={min_n}, edges={G_real.number_of_edges()})')
    add_colorstamp(axes[0, 0])

    draw_network(axes[0, 1], G_shuf, pos_shuf, nl_s, xy_s, sz_s,
                 pos_colors[nl_s],
                 f'Shuffled (n={min_n}, edges={G_shuf.number_of_edges()})')
    add_colorstamp(axes[0, 1])

    # Row 2: speed color
    speed_r = np.array([speed[nd] for nd in nl_r])
    speed_s = np.array([speed[nd] for nd in nl_s])
    vmax_speed = np.percentile(speed, 95)

    draw_network(axes[1, 0], G_real, pos_real, nl_r, xy_r, sz_r,
                 speed_r, 'Real (speed)', cmap='RdYlBu_r',
                 vmin=0, vmax=vmax_speed, cbar_label='cm/s')
    draw_network(axes[1, 1], G_shuf, pos_shuf, nl_s, xy_s, sz_s,
                 speed_s, 'Shuffled (speed)', cmap='RdYlBu_r',
                 vmin=0, vmax=vmax_speed, cbar_label='cm/s')

    # Row 3 left: trajectory
    ax = axes[2, 0]
    ax.scatter(x_pos, y_pos, c=pos_colors, s=4, alpha=0.5, edgecolors='none')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title('Trajectory')
    ax.set_aspect('equal')

    # Row 3 right: metrics comparison
    ax = axes[2, 1]
    metric_names = ['Position R$^2$', 'Dist. corr.', 'Disp. ratio']
    vals_real = [m_real['R2'], m_real['dist_corr'], m_real['disp_ratio']]
    vals_shuf = [m_shuf['R2'], m_shuf['dist_corr'], m_shuf['disp_ratio']]

    x_bar = np.arange(len(metric_names))
    w = 0.35
    bars_r = ax.bar(x_bar - w / 2, vals_real, w, label='Real',
                    color='#2CA02C', edgecolor='black', linewidth=0.5)
    bars_s = ax.bar(x_bar + w / 2, vals_shuf, w, label='Shuffled',
                    color='#FF7F0E', edgecolor='black', linewidth=0.5,
                    alpha=0.6, hatch='//')

    for bar, val in zip(list(bars_r) + list(bars_s),
                        vals_real + vals_shuf):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    ax.set_xticks(x_bar)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_title('Spatial encoding: real vs shuffled')
    ax.legend(fontsize=10)
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)

    fig.suptitle(f'{args.session}: Population recurrence network\n'
                 f'{n_neurons} neurons, k={args.k}, tau=exp_fit, m=3, '
                 f'thr={threshold}',
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = OUT / 'fig_recurrence_real_vs_shuffled.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {path}')
    print(f'Total: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
