#!/usr/bin/env python
"""
Recurrence analysis of hippocampal population activity (NOF data).

Main idea: communities in the population recurrence graph = behavioral regimes.
Uses DRIADA API throughout: Experiment -> MultiTimeSeries -> population_recurrence_graph.

Pipeline:
  1. Load NOF session via DRIADA (Experiment object)
  2. Downsample + smooth calcium, build MultiTimeSeries
  3. population_recurrence_graph(method='mean') — builds all per-neuron RGs
     (with auto tau/dim estimation) and combines into mean recurrence matrix.
     Per-neuron results cached on ts_list entries.
  4. Extract per-neuron tau/dim/RQA from cached graphs
  5. Binarize mean matrix -> SpectralClustering -> time-point communities
  6. Align communities with behavior (speed, rest) -> ANOVA
  7. Windowed RQA on binarized JRP vs speed
  8. Network of Networks: Jaccard on cached per-neuron graphs -> Louvain
  9. Shuffle control via exp.get_multicell_shuffled_calcium()

Usage:
    python run_nof_recurrence.py
    python run_nof_recurrence.py --session NOF_H01_1D --ds 5 --k 10
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
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# DRIADA
# ---------------------------------------------------------------------------
DRIADA_ROOT = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA_ROOT / 'src'))
sys.path.insert(0, str(DRIADA_ROOT / 'tools'))

from load_synchronized_experiments import load_experiment_from_npz
from driada.information.info_base import MultiTimeSeries
from driada.recurrence.population import pairwise_jaccard_sparse
from driada.recurrence.rqa import compute_rqa
from driada.network import Network

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NOF_DATA_DIR = DRIADA_ROOT / 'DRIADA data' / 'NOF' / 'SynchronizedData26_v1'
LNOF_DATA_DIR = DRIADA_ROOT / 'DRIADA data' / 'LNOF' / 'aligned'
OUT_DIR = Path(__file__).parent / 'results'


def _get_npz_path(session):
    """Resolve session name to npz path (NOF or LNOF)."""
    if session.startswith('LNOF'):
        return LNOF_DATA_DIR / f'{session}_aligned.npz'
    return NOF_DATA_DIR / f'{session}_aligned.npz'


def list_nof_sessions():
    """List all available NOF sessions."""
    return sorted([p.stem.replace('_aligned', '')
                   for p in NOF_DATA_DIR.glob('NOF_*_aligned.npz')])


def parse_args():
    p = argparse.ArgumentParser(description='NOF recurrence analysis')
    p.add_argument('--session', default='NOF_H01_1D',
                   help='Session name (or "all" for batch)')
    p.add_argument('--batch', action='store_true',
                   help='Run on matching sessions (use --session as filter, e.g. NOF_H01)')
    p.add_argument('--batch-cache-only', action='store_true',
                   help='Batch: only build and cache mean matrices, skip plots')
    p.add_argument('--ds', type=int, default=5, help='Downsample (default 5 -> ~4 fps)')
    p.add_argument('--k', type=int, default=10, help='k-NN for recurrence graph')
    p.add_argument('--smooth-sigma', type=float, default=2.0, help='Gaussian sigma (0=off)')
    p.add_argument('--jrp-threshold', type=float, default=0.02, help='Binarization threshold')
    p.add_argument('--n-clusters', type=int, default=3)
    p.add_argument('--n-jobs', type=int, default=-1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tau-method', default='first_minimum',
                   choices=['first_minimum', 'exponential_fit'])
    p.add_argument('--max-dim', type=int, default=10)
    return p.parse_args()


def savefig(fig, name, out_dir, suffix=''):
    """Save figure. suffix is appended before extension (e.g. '_k30_thr025')."""
    stem, ext = name.rsplit('.', 1)
    path = out_dir / f'{stem}{suffix}.{ext}'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    out = OUT_DIR / args.session
    out.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Step 1. Load experiment via DRIADA
    # ==================================================================
    print(f'\n{"="*60}')
    print(f'  NOF Recurrence Analysis: {args.session}')
    print(f'{"="*60}')

    npz_path = _get_npz_path(args.session)
    exp = load_experiment_from_npz(npz_path, verbose=False)
    print(f'  {exp.signature}')

    # ==================================================================
    # Step 2. Downsample + smooth -> MultiTimeSeries
    # ==================================================================
    calcium_data = exp.calcium.data[:, ::args.ds]
    speed_ds = exp.dynamic_features['speed'].data[::args.ds]
    rest_ds = (exp.dynamic_features['rest'].data[::args.ds]
               if 'rest' in exp.dynamic_features else None)
    n_neurons, n_frames = calcium_data.shape
    fps_eff = 20.0 / args.ds

    if args.smooth_sigma > 0:
        calcium_data = np.array([gaussian_filter1d(c, args.smooth_sigma)
                                 for c in calcium_data])

    mts = MultiTimeSeries(calcium_data, discrete=False)
    print(f'  Neurons: {n_neurons}, Frames: {n_frames} '
          f'(ds={args.ds}, {fps_eff:.1f} fps, {n_frames/fps_eff:.0f}s)')

    # ==================================================================
    # Step 3. Population recurrence graph (mean) via DRIADA
    #         Cached to disk: mean_matrix + per-neuron tau/dim/RQA.
    #         Reused across runs with different thresholds.
    # ==================================================================
    cache_file = out / f'mean_matrix_ds{args.ds}_k{args.k}_{args.tau_method[:3]}_md{args.max_dim}.npz'

    if cache_file.exists():
        print(f'\n--- Step 3: Loading cached mean matrix ---')
        cached = np.load(cache_file, allow_pickle=True)
        mean_matrix = cached['mean_matrix']
        taus = cached['taus']
        dims = cached['dims']
        if 'rqa_dicts' in cached:
            rqa_dicts = cached['rqa_dicts'].item()
        else:
            # Cache from separate builder (no per-neuron RQA)
            measures = ['DET', 'LAM', 'ENTR', 'L_mean', 'L_max', 'TT']
            rqa_dicts = {m: np.full(n_neurons, np.nan) for m in measures}
        min_n = mean_matrix.shape[0]
        print(f'  Loaded: {min_n} time points, {n_neurons} neurons')
    else:
        print(f'\n--- Step 3: Population recurrence graph (mean, k={args.k}) ---')
        t3 = time.time()

        pop_rg = mts.population_recurrence_graph(
            method='mean', k=args.k, n_jobs=args.n_jobs,
            tau_method=args.tau_method, max_dim=args.max_dim)

        mean_matrix = pop_rg.adj.toarray() if sp.issparse(pop_rg.adj) else pop_rg.adj
        min_n = mean_matrix.shape[0]
        print(f'  Population graph: {min_n} time points, {time.time()-t3:.1f}s')

        # Extract per-neuron tau/dim/RQA from cached graphs
        print(f'  Extracting per-neuron stats...')
        taus = np.zeros(n_neurons, dtype=int)
        dims = np.zeros(n_neurons, dtype=int)
        rqa_list = []
        for i, ts in enumerate(mts.ts_list):
            _, tau_i = ts._recurrence_tau
            _, m_i = ts._recurrence_embedding_dim
            taus[i] = tau_i
            dims[i] = m_i
            _, rg = ts._recurrence_graph_cache
            rqa_list.append(rg.rqa())

        measures = ['DET', 'LAM', 'ENTR', 'L_mean', 'L_max', 'TT']
        rqa_dicts = {m: np.array([r[m] for r in rqa_list]) for m in measures}

        # Save cache
        np.savez_compressed(cache_file, mean_matrix=mean_matrix,
                            taus=taus, dims=dims, rqa_dicts=rqa_dicts)
        print(f'  Cached: {cache_file}')

    median_tau = int(np.median(taus))
    median_dim = int(np.median(dims))
    measures = ['DET', 'LAM', 'ENTR', 'L_mean', 'L_max', 'TT']

    print(f'\n--- Step 4: Per-neuron stats ---')
    print(f'  Tau:  mean={taus.mean():.1f} +/- {taus.std():.1f} [{taus.min()}, {taus.max()}]')
    print(f'  Dim:  mean={dims.mean():.1f} +/- {dims.std():.1f} [{dims.min()}, {dims.max()}]')
    for m in ['DET', 'LAM', 'ENTR']:
        v = rqa_dicts[m]
        print(f'  {m}: {v.mean():.3f} +/- {v.std():.3f}')

    # Save per-neuron results (lightweight, for external use)
    kw = dict(taus=taus, dims=dims)
    for m in measures:
        kw[f'rqa_{m}'] = rqa_dicts[m]
    np.savez_compressed(out / 'per_neuron.npz', **kw)

    # Fig 1: tau/dim distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(taus, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('tau (samples)')
    ax1.set_ylabel('Count')
    ax1.axvline(np.median(taus), color='red', ls='--',
                label=f'median={np.median(taus):.0f}')
    ax1.legend()
    ax2.hist(dims, bins=range(1, 12), edgecolor='black', alpha=0.7, align='left')
    ax2.set_xlabel('Embedding dim m')
    ax2.set_ylabel('Count')
    ax2.axvline(np.median(dims), color='red', ls='--',
                label=f'median={np.median(dims):.0f}')
    ax2.legend()
    fig.suptitle(f'{args.session}: tau and dim (n={n_neurons})')
    fig.tight_layout()
    savefig(fig, 'fig01_tau_dim.png', out)

    # Fig 2: Population mean recurrence heatmap
    median_tau = int(np.median(taus))
    diag_band = (np.abs(np.arange(min_n)[:, None] - np.arange(min_n)[None, :])
                 < median_tau * 3)
    mean_display = mean_matrix.copy()
    mean_display[diag_band] = 0

    vmax = (np.percentile(mean_display[mean_display > 0], 99)
            if np.any(mean_display > 0) else 1)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mean_display, cmap='hot', aspect='equal', origin='lower',
                   vmin=0, vmax=vmax, interpolation='none')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Time index')
    ax.set_title(f'Population recurrence ({n_neurons} neurons, mean)')
    plt.colorbar(im, ax=ax, label='Fraction of recurring neurons', shrink=0.8)
    savefig(fig, 'fig02_population_recurrence.png', out)

    # ==================================================================
    # Step 5. Binarize mean matrix -> community detection in JRP
    # ==================================================================
    threshold = args.jrp_threshold
    # Suffix for parameter-dependent figures
    thr_str = f'{threshold:.3f}'.replace('.', '')
    sfx = f'_k{args.k}_thr{thr_str}'
    print(f'\n--- Step 5: JRP communities (threshold={threshold}) ---')

    jrp_dense = mean_display.copy()  # diagonal already masked
    jrp_binary = (jrp_dense >= threshold).astype(float)
    nnz = int(jrp_binary.sum())
    n_offdiag = min_n * (min_n - 1)
    rr = nnz / n_offdiag if n_offdiag > 0 else 0
    print(f'  nnz={nnz}, RR={rr:.4f}')

    # Auto-lower threshold if too sparse
    if nnz < 100:
        for thr in [0.01, 0.005, 0.002, 0.001]:
            jrp_binary = (jrp_dense >= thr).astype(float)
            nnz = int(jrp_binary.sum())
            if nnz >= 100:
                threshold = thr
                rr = nnz / n_offdiag
                print(f'  Adjusted: threshold={thr}, nnz={nnz}, RR={rr:.4f}')
                break

    jrp_sparse = sp.csr_matrix(jrp_binary)
    jrp_rqa = compute_rqa(jrp_sparse)
    print(f'  DET={jrp_rqa["DET"]:.3f}, LAM={jrp_rqa["LAM"]:.3f}, '
          f'ENTR={jrp_rqa["ENTR"]:.3f}')

    # Spectral clustering on JRP -> time-point communities
    from sklearn.cluster import SpectralClustering

    n_cl = args.n_clusters
    if nnz > 50:
        sc = SpectralClustering(n_clusters=n_cl, affinity='precomputed',
                                random_state=args.seed, n_init=20)
        labels = sc.fit_predict(jrp_binary)
    else:
        print('  WARNING: JRP too sparse, random labels')
        labels = rng.integers(0, n_cl, size=min_n)

    for cl in range(n_cl):
        print(f'  Cluster {cl}: {(labels==cl).sum()} pts '
              f'({100*(labels==cl).mean():.1f}%)')

    # ==================================================================
    # Step 6. Align clusters with behavior
    # ==================================================================
    # Embedding shortens time series; trim_to_min uses the shortest one,
    # so offset = n_frames - min_n (points lost from the longest embedding)
    offset = n_frames - min_n
    speed_aligned = speed_ds[offset:offset + min_n]
    time_aligned = np.arange(min_n) / fps_eff

    if len(speed_aligned) < min_n:
        speed_aligned = np.pad(speed_aligned, (0, min_n - len(speed_aligned)),
                               mode='edge')
    rest_aligned = None
    if rest_ds is not None:
        rest_aligned = rest_ds[offset:offset + min_n]
        if len(rest_aligned) < min_n:
            rest_aligned = np.pad(rest_aligned, (0, min_n - len(rest_aligned)),
                                  mode='edge')

    print(f'\n--- Step 6: Cluster-behavior correspondence ---')
    for cl in range(n_cl):
        m = labels == cl
        print(f'  Cluster {cl}: <speed>={speed_aligned[m].mean():.2f}, n={m.sum()}')

    groups = [speed_aligned[labels == cl] for cl in range(n_cl)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        F, p = stats.f_oneway(*groups)
        ss_b = sum(len(g) * (g.mean() - speed_aligned.mean())**2 for g in groups)
        ss_t = ((speed_aligned - speed_aligned.mean())**2).sum()
        eta2 = ss_b / ss_t if ss_t > 0 else 0
        print(f'  ANOVA: F={F:.2f}, p={p:.2e}, eta2={eta2:.3f}')

    # Fig 3: Clusters vs speed (MAIN RESULT)
    colors = plt.cm.Set1(np.linspace(0, 1, n_cl))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    for cl in range(n_cl):
        m = labels == cl
        ax.scatter(time_aligned[m], speed_aligned[m], c=[colors[cl]], s=2,
                   alpha=0.5, label=f'Cluster {cl}')
    ax.set_ylabel('Speed')
    ax.set_title(f'JRP clusters vs speed ({args.session})')
    ax.legend(markerscale=5)

    ax = axes[1]
    if rest_aligned is not None:
        ax.fill_between(time_aligned, 0, rest_aligned, alpha=0.3, color='gray',
                        label='Rest')
    for cl in range(n_cl):
        m = labels == cl
        ax.scatter(time_aligned[m], np.full(m.sum(), cl + 1) * 0.3,
                   c=[colors[cl]], s=2, alpha=0.5)
    ax.set_ylabel('Rest / clusters')

    ax = axes[2]
    ax.scatter(time_aligned, labels, c=[colors[l] for l in labels], s=2, alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cluster')
    ax.set_yticks(range(n_cl))
    fig.tight_layout()
    savefig(fig, 'fig03_clusters_vs_behavior.png', out, sfx)

    # Fig 4: Speed distributions per cluster
    fig, ax = plt.subplots(figsize=(8, 5))
    for cl in range(n_cl):
        ax.hist(speed_aligned[labels == cl], bins=30, alpha=0.5,
                color=colors[cl], label=f'Cluster {cl}', density=True)
    ax.set_xlabel('Speed')
    ax.set_ylabel('Density')
    ax.set_title('Speed distributions by JRP cluster')
    ax.legend()
    savefig(fig, 'fig04_speed_distributions.png', out, sfx)

    # ==================================================================
    # Step 7. Windowed population RQA vs speed
    # ==================================================================
    print(f'\n--- Step 7: Windowed RQA ---')
    win_size = int(25 * fps_eff)  # ~25 sec
    step = win_size // 2
    n_windows = (min_n - win_size) // step + 1

    win_det = np.full(n_windows, np.nan)
    win_lam = np.full(n_windows, np.nan)
    win_speed = np.full(n_windows, np.nan)
    win_time = np.full(n_windows, np.nan)

    for wi in range(n_windows):
        i0 = wi * step
        i1 = i0 + win_size
        win_adj = jrp_sparse[i0:i1, i0:i1]
        if win_adj.nnz > 10:
            rqa_w = compute_rqa(win_adj)
            win_det[wi] = rqa_w['DET']
            win_lam[wi] = rqa_w['LAM']
        win_speed[wi] = speed_aligned[i0:i1].mean()
        win_time[wi] = time_aligned[i0 + win_size // 2]

    valid = ~np.isnan(win_det) & ~np.isnan(win_speed)
    r_det = r_lam = np.nan
    if valid.sum() > 3:
        r_det, p_det = stats.pearsonr(win_speed[valid], win_det[valid])
        r_lam, p_lam = stats.pearsonr(win_speed[valid], win_lam[valid])
        print(f'  DET vs speed: r={r_det:.3f}, p={p_det:.3e}')
        print(f'  LAM vs speed: r={r_lam:.3f}, p={p_lam:.3e}')

    # Fig 5: Windowed DET/LAM vs speed
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(win_time, win_speed, 'k-', alpha=0.7, label='Speed (mean)')
    axes[0].set_ylabel('Speed')
    axes[0].set_title(f'Windowed analysis ({args.session})')
    axes[0].legend()
    lbl_d = f'DET (r={r_det:.2f})' if not np.isnan(r_det) else 'DET'
    lbl_l = f'LAM (r={r_lam:.2f})' if not np.isnan(r_lam) else 'LAM'
    axes[1].plot(win_time, win_det, 'b-', alpha=0.7, label=lbl_d)
    axes[1].plot(win_time, win_lam, 'r-', alpha=0.7, label=lbl_l)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('RQA')
    axes[1].legend()
    fig.tight_layout()
    savefig(fig, 'fig05_windowed_rqa_vs_speed.png', out, sfx)

    # ==================================================================
    # Step 8. Network of Networks (Jaccard) - additional
    #         Only available when per-neuron graphs were built (not from cache)
    # ==================================================================
    has_per_neuron = hasattr(mts.ts_list[0], '_recurrence_graph_cache') and \
                     mts.ts_list[0]._recurrence_graph_cache is not None
    if not has_per_neuron:
        print(f'\n--- Step 8: Skipped (per-neuron graphs not in memory, loaded from cache) ---')
    else:
        print(f'\n--- Step 8: Network of Networks (Jaccard) ---')
        t8 = time.time()
        per_neuron_graphs = [ts._recurrence_graph_cache[1] for ts in mts.ts_list]
        sim_matrix, jac_mask = pairwise_jaccard_sparse(per_neuron_graphs)
        n_jac = sim_matrix.shape[0]
        upper = sim_matrix[np.triu_indices(n_jac, k=1)]
        print(f'  Jaccard: mean={upper.mean():.4f}, std={upper.std():.4f}')
        print(f'  Time: {time.time()-t8:.1f}s')

        thr_95 = np.percentile(upper, 95)
        sim_thr = sim_matrix.copy()
        sim_thr[sim_thr < thr_95] = 0
        np.fill_diagonal(sim_thr, 0)
        net = Network(adj=sp.csr_matrix(sim_thr), preprocessing='giant_cc',
                      create_nx_graph=True)

        import networkx.algorithms.community as nx_comm
        communities = nx_comm.louvain_communities(net.graph, weight='weight',
                                                  seed=args.seed)
        communities = sorted(communities, key=len, reverse=True)
        print(f'  Nodes: {net.n}, Communities: {len(communities)} '
              f'(sizes: {", ".join(str(len(c)) for c in communities)})')

        # Fig 6: Jaccard similarity matrix reordered by community
        node_order = [n for comm in communities for n in sorted(comm)]
        idx = np.array(node_order)
        sim_reord = sim_matrix[np.ix_(idx, idx)]
        np.fill_diagonal(sim_reord, 0)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(sim_reord, cmap='hot', aspect='auto',
                       vmin=0, vmax=np.percentile(upper, 99))
        cumsum = np.cumsum([0] + [len(c) for c in communities])
        for b in cumsum[1:-1]:
            ax.axhline(b - 0.5, color='cyan', lw=1, alpha=0.8)
            ax.axvline(b - 0.5, color='cyan', lw=1, alpha=0.8)
        ax.set_xlabel('Neuron (reordered)')
        ax.set_ylabel('Neuron')
        ax.set_title(f'Jaccard similarity ({n_neurons} neurons, '
                     f'{len(communities)} communities)')
        plt.colorbar(im, ax=ax, label='Jaccard index')
        savefig(fig, 'fig06_jaccard_similarity.png', out)

    # ==================================================================
    # Step 9. Shuffle control
    # ==================================================================
    print(f'\n--- Step 9: Shuffle control ---')
    t9 = time.time()

    shuf_cache = out / f'mean_matrix_shuffled_ds{args.ds}_k{args.k}_{args.tau_method[:3]}_md{args.max_dim}.npz'
    if shuf_cache.exists():
        print(f'  Loading cached shuffled mean matrix...')
        mean_shuf = np.load(shuf_cache)['mean_matrix']
    else:
        # DRIADA built-in circular shift shuffle
        mts_shuf = exp.get_multicell_shuffled_calcium(return_array=False)
        shuf_data = mts_shuf.data[:, ::args.ds]
        if args.smooth_sigma > 0:
            shuf_data = np.array([gaussian_filter1d(c, args.smooth_sigma)
                                  for c in shuf_data])
        mts_shuf = MultiTimeSeries(shuf_data, discrete=False)

        pop_shuf = mts_shuf.population_recurrence_graph(
            method='mean', k=args.k, n_jobs=args.n_jobs,
            tau_method=args.tau_method, max_dim=args.max_dim)
        mean_shuf = (pop_shuf.adj.toarray() if sp.issparse(pop_shuf.adj)
                     else pop_shuf.adj)
        np.savez_compressed(shuf_cache, mean_matrix=mean_shuf)
        print(f'  Cached: {shuf_cache}')

    # Mask diagonal band
    min_n_s = min(mean_shuf.shape[0], min_n)
    mean_shuf = mean_shuf[:min_n_s, :min_n_s]
    diag_band_s = (np.abs(np.arange(min_n_s)[:, None] - np.arange(min_n_s)[None, :])
                   < median_tau * 3)
    mean_shuf[diag_band_s] = 0

    real_offdiag = mean_display[:min_n_s, :min_n_s][~diag_band_s]
    shuf_offdiag = mean_shuf[~diag_band_s]
    print(f'  Real mean recurrence:     {real_offdiag.mean():.5f}')
    print(f'  Shuffled mean recurrence: {shuf_offdiag.mean():.5f}')

    # RQA comparison: structure of population recurrence, not just mean level
    jrp_shuf_binary = (mean_shuf >= threshold).astype(float)
    jrp_shuf_sparse = sp.csr_matrix(jrp_shuf_binary)
    jrp_shuf_rqa = compute_rqa(jrp_shuf_sparse)

    # Recompute real JRP RQA on the same size for fair comparison
    jrp_real_trimmed = jrp_sparse[:min_n_s, :min_n_s]
    jrp_real_rqa = compute_rqa(jrp_real_trimmed)

    print(f'\n  RQA comparison (structure, not mean level):')
    print(f'  {"":12s} {"Real":>8s} {"Shuffled":>8s} {"Ratio":>8s}')
    print(f'  {"-"*40}')
    for m in ['DET', 'LAM', 'ENTR']:
        rv = jrp_real_rqa[m]
        sv = jrp_shuf_rqa[m]
        ratio = rv / sv if sv > 0 else float('inf')
        print(f'  {m:12s} {rv:8.3f} {sv:8.3f} {ratio:8.2f}')
    real_nnz = jrp_real_trimmed.nnz
    shuf_nnz = jrp_shuf_sparse.nnz
    print(f'  {"nnz":12s} {real_nnz:8d} {shuf_nnz:8d} {real_nnz/shuf_nnz if shuf_nnz > 0 else 0:8.2f}')

    print(f'  Time: {time.time()-t9:.1f}s')

    # Fig 7: Real vs shuffled heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # Use same vmax for both for fair visual comparison
    vmax_common = np.percentile(
        mean_display[:min_n_s, :min_n_s][mean_display[:min_n_s, :min_n_s] > 0], 99
    ) if np.any(mean_display[:min_n_s, :min_n_s] > 0) else 1
    for ax, mat, title in zip(axes,
                                [mean_display[:min_n_s, :min_n_s], mean_shuf],
                                ['Real', 'Shuffled']):
        im = ax.imshow(mat, cmap='hot', aspect='equal', origin='lower',
                       vmin=0, vmax=vmax_common, interpolation='none')
        ax.set_xlabel('Time index')
        ax.set_ylabel('Time index')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'Population recurrence: real vs shuffled ({args.session})')
    fig.tight_layout()
    savefig(fig, 'fig07_real_vs_shuffled.png', out, sfx)

    # Fig 8: RQA bar comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    rqa_measures = ['DET', 'LAM', 'ENTR']
    real_vals = [jrp_real_rqa[m] for m in rqa_measures]
    shuf_vals = [jrp_shuf_rqa[m] for m in rqa_measures]
    x = np.arange(len(rqa_measures))
    w = 0.35
    ax.bar(x - w/2, real_vals, w, label='Real', color='steelblue')
    ax.bar(x + w/2, shuf_vals, w, label='Shuffled', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(rqa_measures)
    ax.set_ylabel('Value')
    ax.set_title(f'Population JRP structure: real vs shuffled ({args.session})')
    ax.legend()
    savefig(fig, 'fig08_rqa_real_vs_shuffled.png', out, sfx)

    # ==================================================================
    # Step 10. JRP as network graph (spring layout, speed coloring)
    # ==================================================================
    print(f'\n--- Step 10: JRP network visualization ---')
    import networkx as nx
    import matplotlib.colors as mcolors

    # Build graph: nodes = time points, edges = binarized JRP
    G = nx.from_scipy_sparse_array(jrp_sparse)
    print(f'  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')

    # Speed percentile per node
    node_speeds = speed_aligned[:min_n]
    speed_pct = np.zeros(min_n)
    for i in range(min_n):
        speed_pct[i] = (node_speeds <= node_speeds[i]).sum() / min_n

    # ForceAtlas2 layout (much faster than spring for large graphs)
    from fa2_modified import ForceAtlas2
    t10 = time.time()
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
    pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=100)
    print(f'  Layout: {time.time()-t10:.1f}s')

    fig, ax = plt.subplots(figsize=(14, 14))
    cmap = plt.cm.RdYlBu_r  # red=fast, blue=slow

    # Draw edges (thin, transparent)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.02, width=0.3,
                           edge_color='gray')

    # Draw nodes
    node_list = list(G.nodes())
    colors = [speed_pct[n] for n in node_list]
    degrees = dict(G.degree())
    sizes = [3 + 15 * degrees.get(n, 0) / max(degrees.values())
             for n in node_list]
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, ax=ax,
                           node_size=sizes, node_color=colors,
                           cmap=cmap, vmin=0, vmax=1,
                           edgecolors='none', linewidths=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, label='Speed percentile')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Slow', '25%', '50%', '75%', 'Fast'])

    ax.set_title(f'Population recurrence network ({args.session})\n'
                 f'{min_n} time points, {G.number_of_edges()} edges '
                 f'(thr={threshold})', fontsize=13)
    ax.axis('off')
    fig.tight_layout()
    savefig(fig, 'fig09_recurrence_network.png', out, sfx)

    # ==================================================================
    # Save all results
    # ==================================================================
    np.savez_compressed(out / 'population_results.npz',
                        mean_matrix=mean_matrix,
                        jrp_labels=labels,
                        speed_aligned=speed_aligned,
                        time_aligned=time_aligned,
                        jaccard_matrix=sim_matrix if 'sim_matrix' in dir() else np.array([]),
                        win_det=win_det, win_lam=win_lam,
                        win_speed=win_speed, win_time=win_time,
                        threshold=threshold,
                        taus=taus, dims=dims)

    total = time.time() - t0
    print(f'\n{"="*60}')
    print(f'  Done in {total:.1f}s ({total/60:.1f} min)')
    print(f'  Results: {out}/')
    print(f'{"="*60}')


def build_cache(session, args):
    """Build and cache mean matrix + shuffled for one session.

    Returns True if successful, False if failed.
    """
    npz_path = _get_npz_path(session)
    if not npz_path.exists():
        print(f'  SKIP: {npz_path} not found')
        return False

    out = OUT_DIR / session
    out.mkdir(parents=True, exist_ok=True)

    cache_file = out / f'mean_matrix_ds{args.ds}_k{args.k}_{args.tau_method[:3]}_md{args.max_dim}.npz'
    shuf_cache = out / f'mean_matrix_shuffled_ds{args.ds}_k{args.k}_{args.tau_method[:3]}_md{args.max_dim}.npz'

    if cache_file.exists() and shuf_cache.exists():
        print(f'  {session}: already cached')
        return True

    print(f'\n  {session}: building cache...')
    t0 = time.time()

    exp = load_experiment_from_npz(npz_path, verbose=False)
    calcium_data = exp.calcium.data[:, ::args.ds]
    n_neurons = calcium_data.shape[0]

    if args.smooth_sigma > 0:
        calcium_data = np.array([gaussian_filter1d(c, args.smooth_sigma)
                                 for c in calcium_data])

    # Real mean matrix
    if not cache_file.exists():
        mts = MultiTimeSeries(calcium_data, discrete=False)
        pop_rg = mts.population_recurrence_graph(
            method='mean', k=args.k, n_jobs=args.n_jobs,
            tau_method=args.tau_method, max_dim=args.max_dim)
        mean_matrix = pop_rg.adj.toarray() if sp.issparse(pop_rg.adj) else pop_rg.adj

        taus = np.zeros(n_neurons, dtype=int)
        dims = np.zeros(n_neurons, dtype=int)
        rqa_list = []
        measures = ['DET', 'LAM', 'ENTR', 'L_mean', 'L_max', 'TT']
        for i, ts in enumerate(mts.ts_list):
            _, tau_i = ts._recurrence_tau
            _, m_i = ts._recurrence_embedding_dim
            taus[i] = tau_i
            dims[i] = m_i
            _, rg = ts._recurrence_graph_cache
            rqa_list.append(rg.rqa())
        rqa_dicts = {m: np.array([r[m] for r in rqa_list]) for m in measures}

        np.savez_compressed(cache_file, mean_matrix=mean_matrix,
                            taus=taus, dims=dims, rqa_dicts=rqa_dicts)
        print(f'    Real: {mean_matrix.shape[0]} pts, {time.time()-t0:.0f}s')

    # Shuffled mean matrix
    if not shuf_cache.exists():
        t1 = time.time()
        mts_shuf = exp.get_multicell_shuffled_calcium(return_array=False)
        shuf_data = mts_shuf.data[:, ::args.ds]
        if args.smooth_sigma > 0:
            shuf_data = np.array([gaussian_filter1d(c, args.smooth_sigma)
                                  for c in shuf_data])
        mts_shuf = MultiTimeSeries(shuf_data, discrete=False)
        pop_shuf = mts_shuf.population_recurrence_graph(
            method='mean', k=args.k, n_jobs=args.n_jobs,
            tau_method=args.tau_method, max_dim=args.max_dim)
        mean_shuf = pop_shuf.adj.toarray() if sp.issparse(pop_shuf.adj) else pop_shuf.adj
        np.savez_compressed(shuf_cache, mean_matrix=mean_shuf)
        print(f'    Shuffled: {mean_shuf.shape[0]} pts, {time.time()-t1:.0f}s')

    print(f'    Total: {time.time()-t0:.0f}s')
    return True


if __name__ == '__main__':
    args = parse_args()

    if args.batch or args.batch_cache_only:
        sessions = list_nof_sessions()
        print(f'Batch mode: {len(sessions)} sessions, ds={args.ds}, k={args.k}')
        t_total = time.time()
        ok, fail = 0, 0
        for session in sessions:
            try:
                if build_cache(session, args):
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                print(f'  {session}: FAILED - {e}')
                fail += 1
        print(f'\nBatch done: {ok} cached, {fail} failed, '
              f'{time.time()-t_total:.0f}s total')

        if not args.batch_cache_only:
            # Run full analysis on each cached session
            for session in sessions:
                cache = OUT_DIR / session / f'mean_matrix_ds{args.ds}_k{args.k}_{args.tau_method[:3]}_md{args.max_dim}.npz'
                if cache.exists():
                    args.session = session
                    main()
    else:
        main()
