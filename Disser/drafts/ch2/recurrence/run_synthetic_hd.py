#!/usr/bin/env python
"""
Population recurrence analysis on synthetic HEAD DIRECTION population.

Simplest case: all neurons are HD-selective with uniform preferred directions.
The population state is fully determined by one circular variable (head direction).
Expected: PRG reveals circular manifold structure — spectral embedding should
produce a ring, and JRP clusters should map to angular sectors.

Usage:
    python run_synthetic_hd.py
    python run_synthetic_hd.py --n-neurons 200 --duration 600 --k 50
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import time
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy import stats

# ---------------------------------------------------------------------------
# DRIADA
# ---------------------------------------------------------------------------
DRIADA_ROOT = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA_ROOT / 'src'))

from driada.experiment.synthetic import generate_tuned_selectivity_exp
from driada.information.info_base import MultiTimeSeries
from driada.recurrence.rqa import compute_rqa

OUT_DIR = Path(__file__).parent / 'results' / 'synthetic_hd'


def parse_args():
    p = argparse.ArgumentParser(description='HD population recurrence')
    p.add_argument('--n-neurons', type=int, default=200)
    p.add_argument('--duration', type=int, default=600)
    p.add_argument('--fps', type=float, default=5)
    p.add_argument('--kappa', type=float, default=4.0,
                   help='Von Mises concentration (default 4.0)')
    p.add_argument('--k', type=int, default=50, help='k-NN for recurrence')
    p.add_argument('--jrp-threshold', type=float, default=None,
                   help='Binarization threshold (auto if None)')
    p.add_argument('--n-clusters', type=int, default=6)
    p.add_argument('--n-jobs', type=int, default=-1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-cache', action='store_true')
    return p.parse_args()


def savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N = args.n_neurons

    # ==================================================================
    # Step 1. Generate HD population via generate_tuned_selectivity_exp
    # ==================================================================
    print(f'\n{"="*60}')
    print(f'  HD Population Recurrence Analysis')
    print(f'{"="*60}')

    population = [
        {"name": "hd_cells", "count": N, "features": ["head_direction"]},
    ]

    exp = generate_tuned_selectivity_exp(
        population=population,
        duration=args.duration,
        fps=args.fps,
        seed=args.seed,
        baseline_rate=0.05,
        peak_rate=2.0,
        decay_time=2.0,
        calcium_noise=0.02,
        verbose=False,
    )

    calcium = exp.calcium.data
    n_neurons, n_frames = calcium.shape
    fps = args.fps
    hd = exp.dynamic_features['head_direction'].data  # radians, [0, 2π)

    print(f'  Neurons: {n_neurons}, Frames: {n_frames} '
          f'({fps:.0f} fps, {n_frames/fps:.0f}s)')
    print(f'  HD range: [{hd.min():.2f}, {hd.max():.2f}] rad')

    # ==================================================================
    # Step 2. Population recurrence graph
    # ==================================================================
    mts = MultiTimeSeries(calcium, discrete=False)
    tag = f'd{args.duration}_fps{fps:.0f}_k{args.k}_n{N}'
    cache_file = OUT_DIR / f'mean_matrix_{tag}.npz'

    if cache_file.exists() and not args.no_cache:
        print(f'\n--- Loading cached mean matrix ---')
        cached = np.load(cache_file, allow_pickle=True)
        mean_matrix = cached['mean_matrix']
        taus = cached['taus']
        dims = cached['dims']
        min_n = mean_matrix.shape[0]
        print(f'  Loaded: {min_n} time points')
    else:
        print(f'\n--- Building population recurrence graph (k={args.k}) ---')
        t2 = time.time()
        pop_rg = mts.population_recurrence_graph(
            method='mean', k=args.k, n_jobs=args.n_jobs)
        mean_matrix = pop_rg.adj.toarray() if sp.issparse(pop_rg.adj) else pop_rg.adj
        min_n = mean_matrix.shape[0]
        print(f'  {min_n} time points, {time.time()-t2:.1f}s')

        taus = np.zeros(n_neurons, dtype=int)
        dims = np.zeros(n_neurons, dtype=int)
        for i, ts in enumerate(mts.ts_list):
            _, tau_i = ts._recurrence_tau
            _, m_i = ts._recurrence_embedding_dim
            taus[i] = tau_i
            dims[i] = m_i

        np.savez_compressed(cache_file, mean_matrix=mean_matrix,
                            taus=taus, dims=dims)
        print(f'  Cached: {cache_file}')

    median_tau = int(np.median(taus))
    offset = n_frames - min_n
    hd_aligned = hd[offset:offset + min_n]
    time_aligned = np.arange(min_n) / fps

    print(f'  Median tau={median_tau}, median dim={int(np.median(dims))}')

    # Mask diagonal band
    diag_band = (np.abs(np.arange(min_n)[:, None] - np.arange(min_n)[None, :])
                 < median_tau * 3)
    mean_display = mean_matrix.copy()
    mean_display[diag_band] = 0

    # ==================================================================
    # Step 3. Population recurrence heatmap
    # ==================================================================
    vmax = (np.percentile(mean_display[mean_display > 0], 99)
            if np.any(mean_display > 0) else 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mean_display, cmap='hot', aspect='equal', origin='lower',
                   vmin=0, vmax=vmax, interpolation='none')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Time index')
    ax.set_title(f'Population recurrence ({N} HD neurons, mean)')
    plt.colorbar(im, ax=ax, label='Fraction of recurring neurons', shrink=0.8)
    savefig(fig, 'fig01_population_recurrence.png')

    # ==================================================================
    # Step 4. Binarize -> SpectralClustering -> compare with HD
    # ==================================================================
    offdiag = mean_display[~diag_band]
    if args.jrp_threshold is not None:
        threshold = args.jrp_threshold
    else:
        threshold = np.percentile(offdiag[offdiag > 0], 95)

    print(f'\n--- JRP clustering (threshold={threshold:.4f}) ---')
    jrp_binary = (mean_display >= threshold).astype(float)
    jrp_sparse = sp.csr_matrix(jrp_binary)
    nnz = int(jrp_binary.sum())
    rr = nnz / (min_n * (min_n - 1)) if min_n > 1 else 0
    print(f'  nnz={nnz}, RR={rr:.4f}')

    jrp_rqa = compute_rqa(jrp_sparse)
    print(f'  DET={jrp_rqa["DET"]:.3f}, LAM={jrp_rqa["LAM"]:.3f}')

    from sklearn.cluster import SpectralClustering
    n_cl = args.n_clusters

    sc = SpectralClustering(n_clusters=n_cl, affinity='precomputed',
                            random_state=args.seed, n_init=20)
    labels = sc.fit_predict(jrp_binary)

    # ==================================================================
    # Step 5. Spectral embedding of JRP -> ring visualization
    # ==================================================================
    print(f'\n--- Spectral embedding (expect ring) ---')
    from sklearn.manifold import SpectralEmbedding

    se = SpectralEmbedding(n_components=3, affinity='precomputed',
                           random_state=args.seed)
    coords = se.fit_transform(jrp_binary)

    # Circular correlation between embedding angle and true HD
    emb_angle = np.arctan2(coords[:, 1], coords[:, 0])
    # Circular correlation
    from scipy.stats import pearsonr
    r_cos, _ = pearsonr(np.cos(emb_angle), np.cos(hd_aligned))
    r_sin, _ = pearsonr(np.sin(emb_angle), np.sin(hd_aligned))
    circ_r = np.sqrt(r_cos**2 + r_sin**2)
    print(f'  Circular correlation (embedding angle vs HD): {circ_r:.3f}')

    # Fig 2: Spectral embedding colored by HD (main result)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 2D embedding colored by HD
    ax = axes[0]
    sc_plot = ax.scatter(coords[:, 0], coords[:, 1], c=hd_aligned,
                         cmap='hsv', s=3, alpha=0.7)
    ax.set_xlabel('SE dim 1')
    ax.set_ylabel('SE dim 2')
    ax.set_title(f'Spectral embedding (color = HD)\ncirc. corr = {circ_r:.3f}')
    ax.set_aspect('equal')
    plt.colorbar(sc_plot, ax=ax, label='Head direction (rad)')

    # 2D embedding colored by cluster
    ax = axes[1]
    cl_colors = plt.cm.Set1(np.linspace(0, 1, n_cl))
    for cl in range(n_cl):
        m = labels == cl
        ax.scatter(coords[m, 0], coords[m, 1], c=[cl_colors[cl]], s=3,
                   alpha=0.7, label=f'C{cl}')
    ax.set_xlabel('SE dim 1')
    ax.set_ylabel('SE dim 2')
    ax.set_title('Spectral embedding (color = cluster)')
    ax.set_aspect('equal')
    ax.legend(markerscale=5, fontsize=8)

    # HD histogram per cluster (polar)
    ax = axes[2]
    ax = fig.add_subplot(1, 3, 3, projection='polar')
    axes[2].set_visible(False)
    bins = np.linspace(0, 2 * np.pi, 25)
    for cl in range(n_cl):
        hd_cl = hd_aligned[labels == cl]
        counts, _ = np.histogram(hd_cl, bins=bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(centers, counts, color=cl_colors[cl], label=f'C{cl}')
        ax.fill_between(centers, 0, counts, color=cl_colors[cl], alpha=0.2)
    ax.set_title('HD distribution per cluster', pad=20)
    ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.0))
    fig.tight_layout()
    savefig(fig, 'fig02_spectral_embedding.png')

    # ==================================================================
    # Step 6. Clusters vs HD over time
    # ==================================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.scatter(time_aligned, hd_aligned, c=hd_aligned, cmap='hsv', s=2,
               alpha=0.6)
    ax.set_ylabel('Head direction (rad)')
    ax.set_title('Ground-truth HD vs JRP clusters')

    ax = axes[1]
    for cl in range(n_cl):
        m = labels == cl
        ax.scatter(time_aligned[m], np.full(m.sum(), cl), c=[cl_colors[cl]],
                   s=3, alpha=0.6)
    ax.set_ylabel('Cluster')
    ax.set_yticks(range(n_cl))

    ax = axes[2]
    ax.scatter(time_aligned, hd_aligned, c=[cl_colors[l] for l in labels],
               s=2, alpha=0.6)
    ax.set_ylabel('HD (colored by cluster)')
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    savefig(fig, 'fig03_clusters_vs_hd.png')

    # ==================================================================
    # Step 7. Quantify: mean HD per cluster, circular variance
    # ==================================================================
    print(f'\n--- Cluster-HD correspondence ---')
    print(f'  {"Cluster":>8} {"n":>5} {"mean HD":>8} {"circ.var":>8}')
    for cl in range(n_cl):
        hd_cl = hd_aligned[labels == cl]
        # Circular mean
        mean_dir = np.arctan2(np.sin(hd_cl).mean(), np.cos(hd_cl).mean())
        if mean_dir < 0:
            mean_dir += 2 * np.pi
        # Circular variance (1 - resultant length)
        R = np.sqrt(np.sin(hd_cl).mean()**2 + np.cos(hd_cl).mean()**2)
        cv = 1 - R
        print(f'  {cl:>8} {len(hd_cl):>5} {np.degrees(mean_dir):>7.1f}° '
              f'{cv:>8.3f}')

    # ==================================================================
    # Step 8. Shuffle control
    # ==================================================================
    print(f'\n--- Shuffle control ---')
    shuf_calcium = np.empty_like(calcium)
    for i in range(n_neurons):
        shift = rng.integers(1, n_frames)
        shuf_calcium[i] = np.roll(calcium[i], shift)

    mts_shuf = MultiTimeSeries(shuf_calcium, discrete=False)
    pop_shuf = mts_shuf.population_recurrence_graph(
        method='mean', k=args.k, n_jobs=args.n_jobs)
    mean_shuf = pop_shuf.adj.toarray() if sp.issparse(pop_shuf.adj) else pop_shuf.adj

    min_n_s = min(mean_shuf.shape[0], min_n)
    mean_shuf = mean_shuf[:min_n_s, :min_n_s]
    diag_band_s = (np.abs(np.arange(min_n_s)[:, None] - np.arange(min_n_s)[None, :])
                   < median_tau * 3)
    mean_shuf[diag_band_s] = 0

    real_off = mean_display[:min_n_s, :min_n_s][~diag_band_s]
    shuf_off = mean_shuf[~diag_band_s]
    print(f'  Real mean:     {real_off.mean():.5f}')
    print(f'  Shuffled mean: {shuf_off.mean():.5f}')

    jrp_shuf = (mean_shuf >= threshold).astype(float)
    jrp_shuf_rqa = compute_rqa(sp.csr_matrix(jrp_shuf))
    jrp_real_rqa = compute_rqa(jrp_sparse[:min_n_s, :min_n_s])
    for m in ['DET', 'LAM', 'ENTR']:
        rv, sv = jrp_real_rqa[m], jrp_shuf_rqa[m]
        ratio = rv / sv if sv > 0 else float('inf')
        print(f'  {m}: {rv:.3f} vs {sv:.3f} ({ratio:.1f}x)')

    # Fig: real vs shuffled
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    vmax_c = np.percentile(
        mean_display[:min_n_s, :min_n_s][mean_display[:min_n_s, :min_n_s] > 0], 99
    ) if np.any(mean_display[:min_n_s, :min_n_s] > 0) else 1
    for ax, mat, title in zip(axes,
                              [mean_display[:min_n_s, :min_n_s], mean_shuf],
                              ['Real', 'Shuffled']):
        im = ax.imshow(mat, cmap='hot', aspect='equal', origin='lower',
                       vmin=0, vmax=vmax_c, interpolation='none')
        ax.set_xlabel('Time index')
        ax.set_ylabel('Time index')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Population recurrence: real vs shuffled (HD)')
    fig.tight_layout()
    savefig(fig, 'fig04_real_vs_shuffled.png')

    # Shuffled embedding for comparison
    se_shuf = SpectralEmbedding(n_components=2, affinity='precomputed',
                                random_state=args.seed)
    jrp_shuf_binary = (mean_shuf >= threshold).astype(float)
    if jrp_shuf_binary.sum() > 100:
        coords_shuf = se_shuf.fit_transform(jrp_shuf_binary)
        hd_shuf = hd_aligned[:min_n_s]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(coords[:, 0], coords[:, 1], c=hd_aligned,
                        cmap='hsv', s=3, alpha=0.7)
        axes[0].set_title('Real')
        axes[0].set_aspect('equal')
        axes[1].scatter(coords_shuf[:, 0], coords_shuf[:, 1], c=hd_shuf,
                        cmap='hsv', s=3, alpha=0.7)
        axes[1].set_title('Shuffled')
        axes[1].set_aspect('equal')
        fig.suptitle('Spectral embedding: real vs shuffled')
        fig.tight_layout()
        savefig(fig, 'fig05_embedding_real_vs_shuffled.png')

    # ==================================================================
    # Summary
    # ==================================================================
    total = time.time() - t0
    print(f'\n{"="*60}')
    print(f'  SUMMARY')
    print(f'{"="*60}')
    print(f'  {N} HD neurons, {args.duration}s @ {fps:.0f} fps')
    print(f'  k={args.k}, threshold={threshold:.4f}, RR={rr:.4f}')
    print(f'  Circular correlation (SE angle vs HD): {circ_r:.3f}')
    print(f'  DET real/shuffled: {jrp_real_rqa["DET"]:.3f}/{jrp_shuf_rqa["DET"]:.3f}')
    print(f'  Done in {total:.1f}s')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
