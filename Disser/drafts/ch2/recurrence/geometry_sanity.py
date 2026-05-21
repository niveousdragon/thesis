#!/usr/bin/env python
"""Sanity check: spectral embedding of JRP recovers (x,y) layout on one session.

Pipeline:
  1. Load mean_matrix + binarize at 95th percentile -> JRP
  2. Keep largest connected component
  3. SpectralEmbedding (n=2) on JRP as affinity matrix
  4. Load (x, y) trajectory, align by embedding offset
  5. Quantitative tests:
       - Pearson correlation of pairwise distance matrices (Mantel-like)
       - Procrustes disparity after optimal rotation/scaling
  6. Plot: layout colored by x / y / time / speed
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
from scipy import stats
from scipy.spatial import procrustes
from sklearn.manifold import SpectralEmbedding

DRIADA = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA / 'src'))
sys.path.insert(0, str(DRIADA / 'tools'))
from load_synchronized_experiments import load_experiment_from_npz

SESSION = 'NOF_H06_1D'   # large, well-behaved
DS = 5
SUBSAMPLE = 4            # spectral on every 4th point (~887 points for H06)
NOF = Path(r'C:\Users\User\PycharmProjects\driada\DRIADA data/NOF/SynchronizedData26_v1')
HERE = Path(__file__).parent
RESULTS = HERE / 'results'
OUT = RESULTS / 'window_rqa_clustering'


def pairwise_dist(X):
    """Pairwise Euclidean distance matrix as flat upper-triangle vector."""
    D = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(-1))
    iu = np.triu_indices(len(X), k=1)
    return D[iu]


def main():
    t0 = time.time()
    cache = RESULTS / SESSION / f'mean_matrix_ds{DS}_k50_exp_md3.npz'
    cached = np.load(cache, allow_pickle=True)
    mean_matrix = cached['mean_matrix']
    taus = cached['taus']
    median_tau = int(np.median(taus))
    min_n = mean_matrix.shape[0]
    print(f'{SESSION}: min_n={min_n}, median tau={median_tau}')

    # Mask diagonal band, binarize at 95th percentile
    diag = (np.abs(np.arange(min_n)[:, None] - np.arange(min_n)[None, :])
            < median_tau * 3)
    m = mean_matrix.copy()
    m[diag] = 0
    thr = np.percentile(m[m > 0], 95) if np.any(m > 0) else 0.01
    jrp = (m >= thr).astype(float)
    nnz = int(jrp.sum())
    print(f'  threshold={thr:.4f}, RR={nnz/(min_n*(min_n-1)):.4f}, nnz={nnz}')

    # Subsample to ~1000 points (uniform in time)
    idx = np.arange(0, min_n, SUBSAMPLE)
    jrp_s = jrp[np.ix_(idx, idx)]
    n_s = len(idx)
    print(f'  Subsampled to {n_s} points')

    # Restrict to giant connected component (spectral needs connected graph)
    G_sparse = sp.csr_matrix(jrp_s)
    n_cc, cc_labels = sp.csgraph.connected_components(G_sparse, directed=False)
    if n_cc > 1:
        cc_sizes = np.bincount(cc_labels)
        giant = cc_labels == np.argmax(cc_sizes)
        idx = idx[giant]
        jrp_s = jrp[np.ix_(idx, idx)]
        n_s = len(idx)
        print(f'  Graph has {n_cc} components, keeping giant CC: {n_s} points')

    # Spectral embedding (2D)
    print(f'  Computing 2D spectral embedding...')
    se = SpectralEmbedding(n_components=2, affinity='precomputed',
                            random_state=0, n_jobs=-1)
    layout = se.fit_transform(jrp_s)
    print(f'  Embedding done ({time.time()-t0:.1f}s total)')

    # Load behavioral data, align with embedding offset (n_frames - min_n)
    exp = load_experiment_from_npz(NOF / f'{SESSION}_aligned.npz', verbose=False)
    n_frames_full = exp.calcium.data.shape[1] // DS + (
        1 if exp.calcium.data.shape[1] % DS else 0)
    offset_full = n_frames_full - min_n
    x_full = exp.dynamic_features['x'].data[::DS]
    y_full = exp.dynamic_features['y'].data[::DS]
    sp_full = exp.dynamic_features['speed'].data[::DS]
    x_aligned = x_full[offset_full:offset_full + min_n][idx]
    y_aligned = y_full[offset_full:offset_full + min_n][idx]
    speed_aligned = sp_full[offset_full:offset_full + min_n][idx]
    t_aligned = idx / (20 / DS)
    xy = np.column_stack([x_aligned, y_aligned])

    # Quantitative tests
    print('\nQuantitative tests:')
    # Mantel-like Pearson on pairwise distances
    d_layout = pairwise_dist(layout)
    d_phys = pairwise_dist(xy)
    r_mantel, p_mantel = stats.pearsonr(d_layout, d_phys)
    r_spearman, _ = stats.spearmanr(d_layout, d_phys)
    print(f'  Pearson on pairwise distances:  r={r_mantel:+.3f}, p={p_mantel:.2e}')
    print(f'  Spearman on pairwise distances: r={r_spearman:+.3f}')

    # Procrustes disparity
    _, _, disparity = procrustes(xy, layout)
    print(f'  Procrustes disparity (0=perfect, 1=worst): {disparity:.4f}')

    # Shuffled control: same on shuffled mean matrix
    shuf_path = RESULTS / SESSION / f'mean_matrix_shuffled_ds{DS}_k50_exp_md3.npz'
    if shuf_path.exists():
        print('\nShuffled control:')
        shuf = np.load(shuf_path, allow_pickle=True)['mean_matrix']
        ns = min(shuf.shape[0], min_n)
        shuf = shuf[:ns, :ns].copy()
        diag2 = (np.abs(np.arange(ns)[:, None] - np.arange(ns)[None, :])
                 < median_tau * 3)
        shuf[diag2] = 0
        jrp_sh = (shuf >= thr).astype(float)
        idx2 = np.arange(0, ns, SUBSAMPLE)
        jrp_sh_s = jrp_sh[np.ix_(idx2, idx2)]
        n_cc2, cc_lab2 = sp.csgraph.connected_components(
            sp.csr_matrix(jrp_sh_s), directed=False)
        if n_cc2 > 1:
            sz = np.bincount(cc_lab2)
            keep = cc_lab2 == np.argmax(sz)
            idx2 = idx2[keep]
            jrp_sh_s = jrp_sh[np.ix_(idx2, idx2)]
        layout_sh = SpectralEmbedding(n_components=2, affinity='precomputed',
                                       random_state=0,
                                       n_jobs=-1).fit_transform(jrp_sh_s)
        x_sh = x_full[offset_full:offset_full + ns][idx2]
        y_sh = y_full[offset_full:offset_full + ns][idx2]
        xy_sh = np.column_stack([x_sh, y_sh])
        d_l_sh = pairwise_dist(layout_sh)
        d_p_sh = pairwise_dist(xy_sh)
        r_sh, p_sh = stats.pearsonr(d_l_sh, d_p_sh)
        _, _, disp_sh = procrustes(xy_sh, layout_sh)
        print(f'  Pearson on pairwise distances: r={r_sh:+.3f}, p={p_sh:.2e}')
        print(f'  Procrustes disparity:          {disp_sh:.4f}')

    # Figure: 4 panels
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, color, label, cmap in [
        (axes[0], x_aligned, 'x position', 'viridis'),
        (axes[1], y_aligned, 'y position', 'viridis'),
        (axes[2], t_aligned, 'time (s)',   'plasma'),
        (axes[3], speed_aligned, 'speed',  'inferno'),
    ]:
        sc = ax.scatter(layout[:, 0], layout[:, 1], c=color, cmap=cmap,
                        s=6, alpha=0.75, edgecolor='none')
        ax.set_aspect('equal')
        ax.set_title(f'colored by {label}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, shrink=0.85, label=label)

    fig.suptitle(
        f'{SESSION}: spectral embedding of JRP, n={n_s} points\n'
        f'Pearson distances r={r_mantel:+.3f}, Procrustes disparity={disparity:.3f}',
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT / f'geometry_sanity_{SESSION}.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
