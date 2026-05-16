#!/usr/bin/env python
"""
HD population recurrence → network layout colored by head direction.

tau: exponential_fit (TDMI characteristic time)
m:   FNN (standard procedure)
Visualization: ForceAtlas2 layout of binarized JRP, nodes colored by HD.

Usage:
    python run_hd_network.py
    python run_hd_network.py --k 50 --n-neurons 200
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
import matplotlib.colors as mcolors
import numpy as np
import scipy.sparse as sp
import networkx as nx

DRIADA_ROOT = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA_ROOT / 'src'))
sys.path.insert(0, str(DRIADA_ROOT / 'tools'))

from driada.experiment.synthetic import generate_tuned_selectivity_exp
from driada.information.info_base import MultiTimeSeries
from driada.recurrence.rqa import compute_rqa

OUT = Path(__file__).parent / 'results' / 'synthetic_hd'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-neurons', type=int, default=200)
    p.add_argument('--duration', type=int, default=600)
    p.add_argument('--fps', type=float, default=5)
    p.add_argument('--k', type=int, default=50)
    p.add_argument('--theiler', type=int, default=5)
    p.add_argument('--tau-method', default='exponential_fit',
                   choices=['first_minimum', 'exponential_fit'])
    p.add_argument('--max-dim', type=int, default=5)
    p.add_argument('--jrp-threshold', type=float, default=None)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ==================================================================
    # 1. Generate HD population
    # ==================================================================
    print(f'\n{"="*60}')
    print(f'  HD Population Recurrence Network')
    print(f'{"="*60}')

    exp = generate_tuned_selectivity_exp(
        population=[{'name': 'hd', 'count': args.n_neurons,
                     'features': ['head_direction']}],
        duration=args.duration, fps=args.fps, seed=args.seed,
        baseline_rate=0.05, peak_rate=2.0, decay_time=2.0,
        calcium_noise=0.02, verbose=False)

    calcium = exp.calcium.data
    hd_full = exp.dynamic_features['head_direction'].data
    n_neurons, n_frames = calcium.shape
    print(f'  {n_neurons} neurons, {n_frames} frames, {args.fps} fps')

    # ==================================================================
    # 2-3. Population recurrence graph via DRIADA API
    # ==================================================================
    print(f'\n--- Population recurrence graph '
          f'(tau_method={args.tau_method}, max_dim={args.max_dim}, '
          f'k={args.k}, theiler={args.theiler}) ---')
    t2 = time.time()

    mts = MultiTimeSeries(calcium, discrete=False)
    pop_rg = mts.population_recurrence_graph(
        method='mean', k=args.k, n_jobs=-1,
        theiler_window=args.theiler,
        tau_method=args.tau_method, max_dim=args.max_dim,
    )
    mm = pop_rg.adj.toarray() if sp.issparse(pop_rg.adj) else pop_rg.adj
    min_n = mm.shape[0]
    offset = n_frames - min_n
    hd = hd_full[offset:offset + min_n]
    time_ax = np.arange(min_n) / args.fps

    # Extract per-neuron stats from caches
    taus = np.array([ts._recurrence_tau[1] for ts in mts.ts_list])
    dims = np.array([ts._recurrence_graph_cache[0][1] for ts in mts.ts_list])
    emb_wins = taus * (dims - 1)

    print(f'  {min_n} time points, {time.time()-t2:.1f}s')
    print(f'  tau:     median={int(np.median(taus))}, '
          f'range=[{taus.min()}, {taus.max()}]')
    print(f'  dim:     median={int(np.median(dims))}, '
          f'range=[{dims.min()}, {dims.max()}]')
    print(f'  emb_win: median={int(np.median(emb_wins))} samples '
          f'= {np.median(emb_wins)/args.fps:.1f}s')

    # Mask diagonal
    diag = np.abs(np.arange(min_n)[:, None] - np.arange(min_n)[None, :]) < 3
    md = mm.copy()
    md[diag] = 0

    # ==================================================================
    # 4. Binarize
    # ==================================================================
    offdiag = md[~diag]
    if args.jrp_threshold is not None:
        threshold = args.jrp_threshold
    else:
        threshold = np.percentile(offdiag[offdiag > 0], 95)

    jrp_binary = (md >= threshold).astype(float)
    jrp_sparse = sp.csr_matrix(jrp_binary)
    nnz = int(jrp_binary.sum())
    rr = nnz / (min_n * (min_n - 1))
    print(f'  threshold={threshold:.4f}, nnz={nnz}, RR={rr:.4f}')

    jrp_rqa = compute_rqa(jrp_sparse)
    print(f'  DET={jrp_rqa["DET"]:.3f}, LAM={jrp_rqa["LAM"]:.3f}')

    # ==================================================================
    # 4b. Quantitative metrics: does the graph encode HD?
    # ==================================================================
    print(f'\n--- HD encoding metrics ---')

    # (a) Mean neighbor angular dispersion
    #     For each node, compute circular variance of HD among its neighbors.
    #     Low = neighbors have similar HD = graph encodes HD.
    jrp_csr = jrp_sparse.tocsr()
    circ_vars = []
    for i in range(min_n):
        neighbors = jrp_csr[i].indices
        if len(neighbors) < 2:
            continue
        hd_nb = hd[neighbors]
        R = np.sqrt(np.mean(np.cos(hd_nb))**2 + np.mean(np.sin(hd_nb))**2)
        circ_vars.append(1 - R)
    mean_cv = np.mean(circ_vars)

    # Null: random permutation of HD labels (1000 shuffles)
    rng = np.random.default_rng(args.seed)
    null_cvs = []
    for _ in range(200):
        hd_shuf = rng.permutation(hd)
        cvs_shuf = []
        for i in range(min_n):
            neighbors = jrp_csr[i].indices
            if len(neighbors) < 2:
                continue
            hd_nb = hd_shuf[neighbors]
            R = np.sqrt(np.mean(np.cos(hd_nb))**2 + np.mean(np.sin(hd_nb))**2)
            cvs_shuf.append(1 - R)
        null_cvs.append(np.mean(cvs_shuf))
    null_cvs = np.array(null_cvs)
    p_val = (null_cvs <= mean_cv).mean()

    print(f'  Neighbor circ. variance: {mean_cv:.4f} '
          f'(null: {null_cvs.mean():.4f} ± {null_cvs.std():.4f}, '
          f'p={p_val:.4f})')
    print(f'  Ratio (null/real): {null_cvs.mean()/mean_cv:.2f}x')

    # (b) Spectral embedding circular correlation
    from sklearn.manifold import SpectralEmbedding
    from scipy.stats import pearsonr as _pearsonr

    se = SpectralEmbedding(n_components=4, affinity='precomputed',
                           random_state=args.seed)
    coords_se = se.fit_transform(md)
    best_circ_r = 0
    for d1 in range(4):
        for d2 in range(d1 + 1, 4):
            ea = np.arctan2(coords_se[:, d2], coords_se[:, d1])
            rc, _ = _pearsonr(np.cos(ea), np.cos(hd))
            rs, _ = _pearsonr(np.sin(ea), np.sin(hd))
            cr = np.sqrt(rc**2 + rs**2)
            if cr > best_circ_r:
                best_circ_r = cr
    print(f'  Spectral embedding circ_r: {best_circ_r:.3f}')

    # ==================================================================
    # 5. Build network + ForceAtlas2 layout
    # ==================================================================
    print(f'\n--- Network layout (ForceAtlas2) ---')
    G = nx.from_scipy_sparse_array(jrp_sparse)
    print(f'  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')

    from fa2_modified import ForceAtlas2
    t5 = time.time()
    fa2 = ForceAtlas2(
        outboundAttractionDistribution=True,
        edgeWeightInfluence=1.0,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,
        verbose=False,
    )
    pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
    print(f'  Layout: {time.time()-t5:.1f}s')

    # ==================================================================
    # 6. Plot: network colored by HD
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # --- Left: colored by HD (hsv) ---
    ax = axes[0]
    cmap = plt.cm.hsv

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.01, width=0.2,
                           edge_color='gray')

    node_list = list(G.nodes())
    node_hd = np.array([hd[n] for n in node_list])
    # Normalize to [0, 1] for hsv
    node_colors = node_hd / (2 * np.pi)

    degrees = dict(G.degree())
    sizes = np.array([3 + 12 * degrees.get(n, 0) / max(degrees.values())
                      for n in node_list])

    nx.draw_networkx_nodes(G, pos, nodelist=node_list, ax=ax,
                           node_size=sizes, node_color=node_colors,
                           cmap=cmap, vmin=0, vmax=1,
                           edgecolors='none', linewidths=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 2*np.pi))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='Head direction (rad)')
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    ax.set_title(f'Population recurrence network (color = HD)\n'
                 f'{min_n} time points, {G.number_of_edges()} edges, '
                 f'τ=exp_fit(med={int(np.median(taus))}), '
                 f'm=FNN(med={int(np.median(dims))})')
    ax.axis('off')

    # --- Right: colored by time ---
    ax = axes[1]
    cmap_t = plt.cm.viridis
    node_time = np.array([n for n in node_list]) / min_n

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.01, width=0.2,
                           edge_color='gray')
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, ax=ax,
                           node_size=sizes, node_color=node_time,
                           cmap=cmap_t, vmin=0, vmax=1,
                           edgecolors='none', linewidths=0)

    sm_t = plt.cm.ScalarMappable(cmap=cmap_t, norm=mcolors.Normalize(0, 1))
    sm_t.set_array([])
    plt.colorbar(sm_t, ax=ax, shrink=0.6, label='Normalized time')
    ax.set_title('Same network (color = time)')
    ax.axis('off')

    fig.tight_layout()
    path = OUT / 'fig_hd_network.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {path}')

    # ==================================================================
    # 7. Heatmap sorted by HD (for reference)
    # ==================================================================
    order = np.argsort(hd)
    mm_sorted = md[np.ix_(order, order)]
    vmax = np.percentile(md[md > 0], 99) if np.any(md > 0) else 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(md, cmap='hot', origin='lower', vmin=0, vmax=vmax)
    axes[0].set_title('Mean matrix (time order)')
    im = axes[1].imshow(mm_sorted, cmap='hot', origin='lower', vmin=0, vmax=vmax)
    axes[1].set_title('Mean matrix (sorted by HD)')
    plt.colorbar(im, ax=axes[1], shrink=0.8)
    fig.suptitle(f'τ=exp_fit(med={int(np.median(taus))}), '
                 f'm=FNN(med={int(np.median(dims))}), '
                 f'win={int(np.median(emb_wins))}')
    fig.tight_layout()
    path2 = OUT / 'fig_hd_matrix.png'
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path2}')

    print(f'\n  Total: {time.time()-t_total:.0f}s')


if __name__ == '__main__':
    main()
