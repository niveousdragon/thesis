#!/usr/bin/env python
"""
Proper geometry metrics for FA2 layout vs (x, y):
  - Mantel (Pearson + Spearman on pairwise distances) — invariant to rotation
  - Procrustes disparity — invariant to rotation, scaling, reflection
  - KNN decoder error (cm) — interpretable physical accuracy

Compares real vs shuffled with matched-density binarization for fairness.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy import stats
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from sklearn.neighbors import KNeighborsRegressor

DRIADA = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA / 'src'))
sys.path.insert(0, str(DRIADA / 'tools'))
from load_synchronized_experiments import load_experiment_from_npz

SESSION = 'NOF_H32_1D'
DS = 5
TARGET_DEGREE = 12
N_SUBSAMPLE = 2000   # for Mantel/Procrustes pdist (full ~3500 too big)
KNN = 10
NOF = DRIADA / 'DRIADA data/NOF/SynchronizedData26_v1'
HERE = Path(__file__).parent
RESULTS = HERE / 'results'
OUT = RESULTS / 'window_rqa_clustering'


def threshold_for_degree(mm, median_tau, target_k):
    """Find threshold giving average degree ~target_k."""
    n = mm.shape[0]
    diag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < median_tau * 3
    m = mm.copy(); m[diag] = 0
    target_edges = int(target_k * n / 2)
    flat = m[m > 0].ravel()
    if len(flat) <= target_edges:
        return 0.0, m
    # threshold = value of (n_offdiag - 2*target_edges) ranked element
    n_keep = 2 * target_edges   # each edge counted twice
    thr = np.sort(flat)[-n_keep]
    return thr, m


def build_fa2_layout(mm, median_tau, target_k, seed=42):
    thr, m = threshold_for_degree(mm, median_tau, target_k)
    jrp = (m >= thr).astype(float)
    jrp_sparse = sp.csr_matrix(jrp)
    G = nx.from_scipy_sparse_array(jrp_sparse)
    print(f'  threshold={thr:.4f}, edges={G.number_of_edges()}, '
          f'<k>={2*G.number_of_edges()/G.number_of_nodes():.1f}')

    from fa2_modified import ForceAtlas2
    fa2 = ForceAtlas2(outboundAttractionDistribution=True,
                      barnesHutOptimize=True, barnesHutTheta=1.2,
                      scalingRatio=2.0, gravity=1.0, verbose=False)
    pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
    layout = np.array([pos[i] for i in range(G.number_of_nodes())])
    return layout, G, jrp_sparse, thr


def compute_metrics(layout, xy, label='', rng_seed=0):
    n = len(layout)
    rng = np.random.default_rng(rng_seed)
    sub = rng.choice(n, min(N_SUBSAMPLE, n), replace=False)
    L = layout[sub]; P = xy[sub]
    # Mantel
    dL = pdist(L); dP = pdist(P)
    r_pear, p_pear = stats.pearsonr(dL, dP)
    r_spear, _ = stats.spearmanr(dL, dP)
    # Procrustes (full n; very fast)
    _, _, disp = procrustes(xy, layout)
    # KNN decoder (cross-validated)
    knn = KNeighborsRegressor(n_neighbors=KNN)
    n_split = n // 2
    perm = rng.permutation(n)
    tr, te = perm[:n_split], perm[n_split:]
    knn.fit(layout[tr], xy[tr])
    pred = knn.predict(layout[te])
    err = np.sqrt(((pred - xy[te])**2).sum(axis=1)).mean()
    print(f'  [{label:>8s}] Mantel Pearson  r={r_pear:+.3f}, p={p_pear:.2e}')
    print(f'  [{label:>8s}] Mantel Spearman r={r_spear:+.3f}')
    print(f'  [{label:>8s}] Procrustes disparity = {disp:.4f}')
    print(f'  [{label:>8s}] KNN decoder error = {err:.2f} cm (k={KNN})')
    return {'mantel_pearson': r_pear, 'mantel_spearman': r_spear,
            'procrustes': disp, 'knn_err': err}


def main():
    print(f'=== Geometry metrics: {SESSION} ===\n')
    cache = RESULTS / SESSION / f'mean_matrix_ds{DS}_k50_exp_md3.npz'
    shuf = RESULTS / SESSION / f'mean_matrix_shuffled_ds{DS}_k50_exp_md3.npz'
    real_mm = np.load(cache, allow_pickle=True)['mean_matrix']
    shuf_mm = np.load(shuf, allow_pickle=True)['mean_matrix']
    taus = np.load(cache, allow_pickle=True)['taus']
    median_tau = int(np.median(taus))
    n = min(real_mm.shape[0], shuf_mm.shape[0])
    real_mm = real_mm[:n, :n]; shuf_mm = shuf_mm[:n, :n]
    print(f'  n={n}, median tau={median_tau}\n')

    # Load (x, y)
    exp = load_experiment_from_npz(NOF / f'{SESSION}_aligned.npz', verbose=False)
    n_full = exp.calcium.data.shape[1] // DS + (
        1 if exp.calcium.data.shape[1] % DS else 0)
    offset = n_full - n
    x = exp.dynamic_features['x'].data[::DS][offset:offset + n]
    y = exp.dynamic_features['y'].data[::DS][offset:offset + n]
    xy = np.column_stack([x, y])

    # Real
    print('Real graph layout:')
    t0 = time.time()
    layout_r, G_r, jrp_r, thr_r = build_fa2_layout(real_mm, median_tau, TARGET_DEGREE)
    print(f'  FA2 done in {time.time()-t0:.1f}s')
    m_r = compute_metrics(layout_r, xy, label='real', rng_seed=42)

    # Shuffled, matched-density (use same number of edges as real)
    print(f'\nShuffled graph (matched-density to real {G_r.number_of_edges()} edges):')
    n_edges_target = G_r.number_of_edges()
    diag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < median_tau * 3
    msh = shuf_mm.copy(); msh[diag] = 0
    flat_sh = msh[msh > 0].ravel()
    if len(flat_sh) > 2 * n_edges_target:
        thr_sh = np.sort(flat_sh)[-2 * n_edges_target]
    else:
        thr_sh = 0.0
    jrp_sh = (msh >= thr_sh).astype(float)
    G_sh = nx.from_scipy_sparse_array(sp.csr_matrix(jrp_sh))
    print(f'  shuffled threshold={thr_sh:.4f}, edges={G_sh.number_of_edges()}, '
          f'<k>={2*G_sh.number_of_edges()/G_sh.number_of_nodes():.1f}')

    from fa2_modified import ForceAtlas2
    t0 = time.time()
    fa2 = ForceAtlas2(outboundAttractionDistribution=True,
                      barnesHutOptimize=True, barnesHutTheta=1.2,
                      scalingRatio=2.0, gravity=1.0, verbose=False)
    pos_sh = fa2.forceatlas2_networkx_layout(G_sh, pos=None, iterations=200)
    print(f'  FA2 done in {time.time()-t0:.1f}s')
    layout_sh = np.array([pos_sh[i] for i in range(G_sh.number_of_nodes())])
    m_sh = compute_metrics(layout_sh, xy, label='shuffled', rng_seed=43)

    # Summary
    print('\n=== Summary ===')
    print(f'  {"Metric":<22} {"Real":>10} {"Shuffled":>10}')
    for k in ['mantel_pearson', 'mantel_spearman', 'procrustes', 'knn_err']:
        print(f'  {k:<22} {m_r[k]:>10.4f} {m_sh[k]:>10.4f}')

    # Random-shuffle baseline for KNN error: predict mean
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    half = n // 2
    pred_random = xy[perm[:half]].mean(0)
    err_random = np.sqrt(((pred_random - xy[perm[half:]])**2).sum(axis=1)).mean()
    print(f'  {"random baseline":<22} {"":>10} {err_random:>10.2f} cm')

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    # Real layout colored by position
    cmap = plt.cm.viridis
    pos_color = (x - x.min()) / max(x.max() - x.min(), 1)
    axes[0].scatter(layout_r[:, 0], layout_r[:, 1], c=x, cmap=cmap,
                    s=4, alpha=0.7, edgecolors='none')
    axes[0].set_title(f'Real (color=x)', fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([]); axes[0].set_aspect('equal')

    axes[1].scatter(layout_r[:, 0], layout_r[:, 1], c=y, cmap=cmap,
                    s=4, alpha=0.7, edgecolors='none')
    axes[1].set_title(f'Real (color=y)', fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].set_aspect('equal')

    axes[2].scatter(layout_sh[:, 0], layout_sh[:, 1], c=x, cmap=cmap,
                    s=4, alpha=0.7, edgecolors='none')
    axes[2].set_title(f'Shuffled, matched density', fontsize=11)
    axes[2].set_xticks([]); axes[2].set_yticks([]); axes[2].set_aspect('equal')

    fig.suptitle(
        f'{SESSION}: real Mantel={m_r["mantel_pearson"]:+.2f}, '
        f'KNN err={m_r["knn_err"]:.1f}cm | '
        f'shuf Mantel={m_sh["mantel_pearson"]:+.2f}, '
        f'KNN err={m_sh["knn_err"]:.1f}cm',
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT / f'geometry_metrics_{SESSION}.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
