#!/usr/bin/env python
"""
Full population recurrence pipeline: compute + visualize + metrics.

Steps:
  1. Load session, build population recurrence graph (exp_fit tau, m<=3)
  2. Build shuffled PRG
  3. Binarize both, build ForceAtlas2 layouts
  4. Compute spatial metrics (R^2, dist corr, dispersion ratio)
  5. Generate figures:
     - fig_network_position.png: real vs shuffled, 2D position colormap
     - fig_network_behaviors.png: real network colored by speed, x, y, zone, state, HD
     - fig_metrics.png: bar chart real vs shuffled

Usage:
    python run_recurrence_full.py --session NOF_H32_1D
    python run_recurrence_full.py --session LNOF_J06_1D --k 20
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
from driada.information.info_base import MultiTimeSeries
from driada.recurrence.rqa import compute_rqa
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

OUT_BASE = Path(__file__).parent / 'results'
NOF_DATA_DIR = DRIADA_ROOT / 'DRIADA data' / 'NOF' / 'SynchronizedData26_v1'
LNOF_DATA_DIR = DRIADA_ROOT / 'DRIADA data' / 'LNOF' / 'aligned'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--session', default='NOF_H32_1D')
    p.add_argument('--ds', type=int, default=5)
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--jrp-threshold', type=float, default=None,
                   help='Manual threshold (overrides --k-graph)')
    p.add_argument('--k-graph', type=float, default=12,
                   help='Target average degree for binarized JRP (default 12)')
    p.add_argument('--smooth-sigma', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-jobs', type=int, default=-1)
    return p.parse_args()


def get_npz_path(session):
    if session.startswith('LNOF'):
        return LNOF_DATA_DIR / f'{session}_aligned.npz'
    return NOF_DATA_DIR / f'{session}_aligned.npz'


def savefig(fig, path):
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ── Graph construction ─────────────────────────────────────────────

def build_prg(calcium, k, n_jobs):
    """Build population recurrence graph with exp_fit tau, m<=3.

    Memory-efficient: builds one graph at a time, sums on the fly.
    Two passes: 1) estimate sizes, 2) build + accumulate.
    """
    from driada.recurrence.embedding import (
        estimate_tau, estimate_embedding_dim, takens_embedding)
    from driada.recurrence.recurrence_graph import RecurrenceGraph

    n_neurons, T = calcium.shape
    taus = np.zeros(n_neurons, dtype=int)
    dims = np.zeros(n_neurons, dtype=int)
    sizes = np.zeros(n_neurons, dtype=int)

    # Pass 1: estimate tau/dim/size (no graph kept)
    print(f'    Pass 1: embedding params ({n_neurons} neurons)...')
    for i in range(n_neurons):
        tau_i = estimate_tau(calcium[i], method='exponential_fit')
        m_i = estimate_embedding_dim(calcium[i], tau=tau_i, max_dim=3)
        taus[i] = tau_i
        dims[i] = m_i
        sizes[i] = T - tau_i * (m_i - 1)

    # Adaptive trim: remove outliers by size
    losses = sizes.max() - sizes
    mean_loss = losses.mean()
    std_loss = losses.std()
    if std_loss > 0:
        kept_mask = losses <= mean_loss + 3.0 * std_loss
    else:
        kept_mask = np.ones(n_neurons, dtype=bool)
    n_removed = n_neurons - kept_mask.sum()
    if n_removed > 0:
        print(f'    Adaptive trim: removed {n_removed}/{n_neurons}')
    min_n = int(sizes[kept_mask].min())

    # Pass 2: build one graph at a time, accumulate into dense array
    n_kept = int(kept_mask.sum())
    mem_gb = min_n**2 * 8 / 1e9
    print(f'    Pass 2: streaming sum ({n_kept} neurons, {min_n} pts, '
          f'accumulator {mem_gb:.1f} GB)...')
    mm = np.zeros((min_n, min_n), dtype=np.float64)
    for i in range(n_neurons):
        if not kept_mask[i]:
            continue
        emb = takens_embedding(calcium[i], tau=int(taus[i]), m=int(dims[i]))
        rg = RecurrenceGraph(emb, method='knn', k=k, theiler_window=5)
        # Add sparse directly into dense (efficient: only touches nnz entries)
        adj = rg.adj[:min_n, :min_n].tocoo()
        mm[adj.row, adj.col] += 1.0
        del rg, adj, emb

    mm /= n_kept
    return mm, taus[kept_mask], dims[kept_mask]


def binarize_and_layout(mm, med_tau, threshold):
    """Mask diagonal, binarize, build graph + FA2 layout."""
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
                      scalingRatio=10.0, gravity=0.1, verbose=False)
    pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
    return G, pos, jrp_sparse


# ── Metrics ────────────────────────────────────────────────────────

def compute_spatial_metrics(G, pos_fa2, jrp_sparse, x, y, n, seed=42):
    rng = np.random.default_rng(seed)
    fa2_xy = np.array([pos_fa2[i] for i in range(n)])
    pos_ref = np.column_stack([x, y])

    # R^2
    r2 = LinearRegression().fit(fa2_xy, pos_ref).score(fa2_xy, pos_ref)

    # Distance correlation
    n_sub = min(2000, n)
    idx = rng.choice(n, n_sub, replace=False)
    rho, _ = spearmanr(pdist(fa2_xy[idx]), pdist(pos_ref[idx]))

    # Neighbor dispersion
    csr = jrp_sparse.tocsr()
    disps = []
    for i in range(n):
        nb = csr[i].indices
        if len(nb) < 2:
            continue
        disps.append(np.sqrt(((x[nb]-x[nb].mean())**2 +
                              (y[nb]-y[nb].mean())**2).mean()))
    mean_disp = np.mean(disps) if disps else np.nan

    null_disps = []
    for _ in range(200):
        p = rng.permutation(n)
        ds = []
        for i in range(n):
            nb = csr[i].indices
            if len(nb) < 2:
                continue
            ds.append(np.sqrt(((x[p][nb]-x[p][nb].mean())**2 +
                               (y[p][nb]-y[p][nb].mean())**2).mean()))
        if ds:
            null_disps.append(np.mean(ds))
    null_mean = np.mean(null_disps) if null_disps else np.nan
    ratio = null_mean / mean_disp if mean_disp > 0 else 0

    return {'R2': r2, 'dist_corr': rho, 'disp_ratio': ratio,
            'disp': mean_disp, 'disp_null': null_mean}


# ── Drawing helpers ────────────────────────────────────────────────

def get_node_data(G, pos):
    nl = list(G.nodes())
    xy = np.array([pos[nd] for nd in nl])
    deg = dict(G.degree())
    mx = max(deg.values()) if deg and max(deg.values()) > 0 else 1
    sz = np.array([10 + 30 * deg.get(nd, 0) / mx for nd in nl])
    return nl, xy, sz


def draw_net(ax, G, pos, nl, xy, sz, colors, title,
             cmap=None, vmin=None, vmax=None, cbar_label=None):
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.008, width=0.2,
                           edge_color='gray')
    if cmap:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap=cmap, s=sz,
                        vmin=vmin, vmax=vmax, edgecolors='none', alpha=0.85,
                        zorder=2)
        if cbar_label:
            plt.colorbar(sc, ax=ax, shrink=0.4, label=cbar_label, pad=0.01)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=sz,
                   edgecolors='none', alpha=0.85, zorder=2)
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def draw_colorstamp(ax):
    ng = 80
    g = np.linspace(0, 1, ng)
    xx, yy = np.meshgrid(g, g)
    pg = np.column_stack([xx.ravel(), yy.ravel()])
    cg = np.zeros((ng*ng, 3))
    cg[:, 0] = pg[:, 0]*(1-pg[:, 1]) + (1-pg[:, 0])*pg[:, 1]
    cg[:, 1] = pg[:, 0]*pg[:, 1] + (1-pg[:, 0])*(1-pg[:, 1])
    cg[:, 2] = pg[:, 1]
    ax_s = inset_axes(ax, width="10%", height="10%", loc="upper right",
                      borderpad=0.3)
    ax_s.imshow(cg.reshape(ng, ng, 3), origin='lower', extent=[0, 1, 0, 1])
    ax_s.set_xticks([0, 1]); ax_s.set_yticks([0, 1])
    ax_s.set_xticklabels(['L', 'R'], fontsize=6)
    ax_s.set_yticklabels(['B', 'T'], fontsize=6)
    ax_s.tick_params(length=0)


def draw_categorical(ax, G, pos, nl, xy, sz, values, color_map, label_map, title):
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.008, width=0.2,
                           edge_color='gray')
    for val in sorted(color_map.keys()):
        mask = values == val
        if mask.any():
            ax.scatter(xy[mask, 0], xy[mask, 1], c=color_map[val], s=sz[mask],
                       edgecolors='none', alpha=0.8, zorder=2,
                       label=label_map.get(val, str(val)))
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    ax.legend(fontsize=8, markerscale=2, loc='lower right')


# ── Main ───────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.time()
    OUT = OUT_BASE / args.session
    OUT.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  {args.session}: Full recurrence pipeline')
    print(f'{"="*60}')

    exp = load_experiment_from_npz(get_npz_path(args.session), verbose=False)
    calcium = exp.calcium.data[:, ::args.ds]
    if args.smooth_sigma > 0:
        calcium = np.array([gaussian_filter1d(c, args.smooth_sigma)
                            for c in calcium])
    n_neurons, n_frames = calcium.shape
    fps = 20.0 / args.ds
    print(f'  {n_neurons} neurons, {n_frames} frames ({fps:.0f} fps)')

    # ── 2. Compute or load PRG ─────────────────────────────────────
    cache_tag = f'ds{args.ds}_k{args.k}_exp_md3'
    real_cache = OUT / f'mean_matrix_{cache_tag}.npz'
    shuf_cache = OUT / f'mean_matrix_shuffled_{cache_tag}.npz'

    # Compute real PRG (cache + free before shuffled to save memory)
    if not real_cache.exists():
        print(f'\n  Computing real PRG (k={args.k})...')
        t1 = time.time()
        mm_real, taus, dims = build_prg(calcium, args.k, args.n_jobs)
        np.savez_compressed(real_cache, mean_matrix=mm_real, taus=taus, dims=dims)
        del mm_real  # free before shuffled
        print(f'  {time.time()-t1:.0f}s, cached: {real_cache.name}')

    # Compute shuffled PRG
    if not shuf_cache.exists():
        print(f'  Computing shuffled PRG...')
        t1 = time.time()
        rng = np.random.default_rng(args.seed)
        shuf_calcium = np.empty_like(calcium)
        for i in range(n_neurons):
            shuf_calcium[i] = np.roll(calcium[i], rng.integers(1, n_frames))
        mm_shuf, _, _ = build_prg(shuf_calcium, args.k, args.n_jobs)
        np.savez_compressed(shuf_cache, mean_matrix=mm_shuf)
        del mm_shuf, shuf_calcium
        print(f'  {time.time()-t1:.0f}s, cached: {shuf_cache.name}')

    # Load both from cache (memory-mapped if possible)
    print(f'\n  Loading cached PRGs...')
    d = np.load(real_cache, allow_pickle=True)
    mm_real, taus, dims = d['mean_matrix'], d['taus'], d['dims']
    mm_shuf = np.load(shuf_cache)['mean_matrix']

    med_tau = int(np.median(taus))
    emb_win = int(np.median(taus * (dims - 1)))
    print(f'  tau={np.median(taus):.0f}, dim={np.median(dims):.0f}, '
          f'emb_win={emb_win} ({emb_win/fps:.1f}s)')

    # Trim to common size
    min_n = min(mm_real.shape[0], mm_shuf.shape[0])
    mm_real = mm_real[:min_n, :min_n]
    mm_shuf = mm_shuf[:min_n, :min_n]
    offset = n_frames - min_n

    # ── 3. Align behavioral variables ──────────────────────────────
    def align(key):
        d = exp.dynamic_features[key].data
        a = d[::args.ds][offset:offset + min_n]
        if len(a) < min_n:
            a = np.pad(a, (0, min_n - len(a)), mode='edge')
        return a

    feat_keys = list(exp.dynamic_features.keys())
    x_pos = align('x')
    y_pos = align('y')
    speed = align('speed')
    pos_colors = position_to_color(np.column_stack([x_pos, y_pos]))

    # Optional features
    has = lambda k: k in feat_keys
    rest = align('rest') if has('rest') else None
    walk = align('walk') if has('walk') else None
    rear = align('rear') if has('rear') else None
    freezing = align('freezing') if has('freezing') else None
    corners = align('corners') if has('corners') else None
    walls = align('walls') if has('walls') else None
    center = align('center') if has('center') else None
    hd = align('headdirection') if has('headdirection') else None

    # ── 4. Build graphs + layouts ──────────────────────────────────
    # Compute threshold from target average degree
    diag_mask = (np.abs(np.arange(min_n)[:, None] - np.arange(min_n)[None, :])
                 < med_tau * 3)
    md_tmp = mm_real.copy()
    md_tmp[diag_mask] = 0

    if args.jrp_threshold is not None:
        threshold = args.jrp_threshold
        print(f'\n  Manual threshold: {threshold:.4f}')
    else:
        # Target: n_edges = min_n * k_graph / 2
        # Each edge counted twice in the symmetric matrix → need n_target nnz entries
        n_target = int(min_n * args.k_graph)
        offdiag_vals = md_tmp[md_tmp > 0]
        if n_target >= len(offdiag_vals):
            threshold = offdiag_vals.min() if len(offdiag_vals) > 0 else 0
        else:
            threshold = np.partition(offdiag_vals, -n_target)[-n_target]
        actual_nnz = (md_tmp >= threshold).sum()
        actual_k = actual_nnz / min_n
        print(f'\n  Target <k>={args.k_graph:.0f} -> threshold={threshold:.5f} '
              f'(actual <k>={actual_k:.1f})')

    def _gc_frac(G):
        if G.number_of_nodes() == 0:
            return 0.0
        gcc = max(nx.connected_components(G), key=len)
        return len(gcc) / G.number_of_nodes()

    print(f'  Building real graph (thr={threshold:.5f})...')
    G_r, pos_r, jrp_r = binarize_and_layout(mm_real, med_tau, threshold)
    gc_r = _gc_frac(G_r)
    print(f'    {G_r.number_of_edges()} edges, <k>={2*G_r.number_of_edges()/G_r.number_of_nodes():.1f}, '
          f'GC={gc_r:.0%}')
    if gc_r < 0.5:
        print(f'    WARNING: giant component < 50%, consider increasing --k-graph')

    print(f'  Building shuffled graph...')
    G_s, pos_s, jrp_s = binarize_and_layout(mm_shuf, med_tau, threshold)
    gc_s = _gc_frac(G_s)
    print(f'    {G_s.number_of_edges()} edges, <k>={2*G_s.number_of_edges()/G_s.number_of_nodes():.1f}, '
          f'GC={gc_s:.0%}')
    print(f'  Edge ratio (real/shuf): {G_r.number_of_edges()/max(G_s.number_of_edges(),1):.1f}x')

    nl_r, xy_r, sz_r = get_node_data(G_r, pos_r)
    nl_s, xy_s, sz_s = get_node_data(G_s, pos_s)

    # ── 5. Metrics ─────────────────────────────────────────────────
    print(f'\n  Spatial metrics (real):')
    m_r = compute_spatial_metrics(G_r, pos_r, jrp_r, x_pos, y_pos,
                                  min_n, args.seed)
    for k, v in m_r.items():
        print(f'    {k}: {v:.3f}')

    print(f'  Spatial metrics (shuffled):')
    m_s = compute_spatial_metrics(G_s, pos_s, jrp_s, x_pos, y_pos,
                                  min_n, args.seed + 1)
    for k, v in m_s.items():
        print(f'    {k}: {v:.3f}')

    # ── 6. Figure 1: real vs shuffled (position color) ─────────────
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    draw_net(axes[0, 0], G_r, pos_r, nl_r, xy_r, sz_r,
             pos_colors[nl_r], f'Real ({G_r.number_of_edges()} edges)')
    draw_colorstamp(axes[0, 0])

    draw_net(axes[0, 1], G_s, pos_s, nl_s, xy_s, sz_s,
             pos_colors[nl_s], f'Shuffled ({G_s.number_of_edges()} edges)')
    draw_colorstamp(axes[0, 1])

    # Trajectory
    axes[1, 0].scatter(x_pos, y_pos, c=pos_colors, s=4, alpha=0.5,
                       edgecolors='none')
    axes[1, 0].set_xlabel('X (cm)')
    axes[1, 0].set_ylabel('Y (cm)')
    axes[1, 0].set_title('Trajectory')
    axes[1, 0].set_aspect('equal')

    # Metrics
    ax = axes[1, 1]
    names = ['Position R$^2$', 'Dist. corr.', 'Disp. ratio']
    vr = [m_r['R2'], m_r['dist_corr'], m_r['disp_ratio']]
    vs = [m_s['R2'], m_s['dist_corr'], m_s['disp_ratio']]
    x_bar = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x_bar - w/2, vr, w, label='Real', color='#2CA02C',
                edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x_bar + w/2, vs, w, label='Shuffled', color='#FF7F0E',
                edgecolor='black', linewidth=0.5, alpha=0.6, hatch='//')
    for b, v in zip(list(b1) + list(b2), vr + vs):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.set_xticks(x_bar)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_title('Spatial encoding: real vs shuffled')
    ax.legend(fontsize=10)
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)

    fig.suptitle(f'{args.session}: Population recurrence network\n'
                 f'{n_neurons} neurons, k={args.k}, tau=exp_fit, m=3, '
                 f'thr={threshold}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, OUT / 'fig_real_vs_shuffled.png')

    # ── 7. Figure 2: behaviors (real network only) ─────────────────
    n_panels = 2  # speed + x/y always available
    panel_data = []

    # Speed
    panel_data.append(('Speed', speed[nl_r], 'RdYlBu_r', 0,
                       np.percentile(speed, 95), 'cm/s'))
    # X
    panel_data.append(('X position', x_pos[nl_r], 'viridis',
                       None, None, 'cm'))
    # Y
    panel_data.append(('Y position', y_pos[nl_r], 'viridis',
                       None, None, 'cm'))

    # Categorical panels
    cat_panels = []

    # Behavioral state
    if rest is not None or walk is not None:
        state = np.zeros(min_n, dtype=int)
        state_colors = {0: '#cccccc'}
        state_labels = {0: 'other'}
        if rest is not None:
            state[rest > 0.5] = 1
            state_colors[1] = '#3366cc'
            state_labels[1] = 'rest'
        if walk is not None:
            state[walk > 0.5] = 2
            state_colors[2] = '#33cc33'
            state_labels[2] = 'walk'
        if rear is not None:
            state[rear > 0.5] = 3
            state_colors[3] = '#ff4444'
            state_labels[3] = 'rear'
        if freezing is not None:
            state[freezing > 0.5] = 4
            state_colors[4] = '#9933cc'
            state_labels[4] = 'freeze'
        cat_panels.append(('Behavioral state', state[nl_r],
                           state_colors, state_labels))

    # Spatial zone
    if corners is not None or walls is not None or center is not None:
        zone = np.zeros(min_n, dtype=int)
        zone_colors = {0: '#cccccc'}
        zone_labels = {0: 'other'}
        if corners is not None:
            zone[corners > 0.5] = 1
            zone_colors[1] = '#cc3333'
            zone_labels[1] = 'corners'
        if walls is not None:
            zone[walls > 0.5] = 2
            zone_colors[2] = '#ff9933'
            zone_labels[2] = 'walls'
        if center is not None:
            zone[center > 0.5] = 3
            zone_colors[3] = '#3366cc'
            zone_labels[3] = 'center'
        cat_panels.append(('Spatial zone', zone[nl_r],
                           zone_colors, zone_labels))

    # HD
    if hd is not None:
        panel_data.append(('Head direction', hd[nl_r], 'hsv',
                           -np.pi, np.pi, 'rad'))

    n_cont = len(panel_data)
    n_cat = len(cat_panels)
    n_total = n_cont + n_cat
    ncols = 3
    nrows = (n_total + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    idx = 0
    for title, c, cmap, vmin, vmax, cbl in panel_data:
        r, col = divmod(idx, ncols)
        draw_net(axes[r, col], G_r, pos_r, nl_r, xy_r, sz_r,
                 c, title, cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=cbl)
        idx += 1

    for title, vals, cm, lm in cat_panels:
        r, col = divmod(idx, ncols)
        draw_categorical(axes[r, col], G_r, pos_r, nl_r, xy_r, sz_r,
                         vals, cm, lm, title)
        idx += 1

    # Hide unused
    while idx < nrows * ncols:
        r, col = divmod(idx, ncols)
        axes[r, col].axis('off')
        idx += 1

    fig.suptitle(f'{args.session}: Network colored by behavior', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, OUT / 'fig_network_behaviors.png')

    # ── Done ───────────────────────────────────────────────────────
    print(f'\n  Total: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
