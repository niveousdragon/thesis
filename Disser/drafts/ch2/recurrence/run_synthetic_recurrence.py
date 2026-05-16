#!/usr/bin/env python
"""
Population recurrence analysis on SYNTHETIC data (ground-truth validation).

Same pipeline as run_nof_recurrence.py, but on a synthetic population from
DRIADA examples where ground-truth module structure and event times are known.

Goal: verify that JRP clusters correspond to behavioral regimes (events),
and that population recurrence metrics distinguish real from shuffled data.

Population (mirroring recurrence_population example):
  - 3 single-event modules (30 neurons each): event_0, event_1, event_2
  - 3 dual-event OR modules (10 neurons each): event_0|1, event_0|2, event_1|2
  Total: 120 neurons, 3 discrete events

Pipeline:
  1. Generate synthetic experiment via DRIADA
  2. Build population recurrence graph (mean method)
  3. Per-neuron tau/dim/RQA
  4. Binarize mean matrix -> SpectralClustering -> time-point communities
  5. Align communities with ground-truth events (confusion matrix, ARI)
  6. Windowed RQA vs event density
  7. Jaccard network -> Louvain -> compare with ground-truth modules (ARI)
  8. Shuffle control (circular shift)

Usage:
    python run_synthetic_recurrence.py
    python run_synthetic_recurrence.py --duration 600 --k 50
    python run_synthetic_recurrence.py --duration 300 --fps 5 --k 30
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

from driada.experiment.synthetic import generate_tuned_selectivity_exp
from driada.information.info_base import MultiTimeSeries
from driada.recurrence.population import pairwise_jaccard_sparse
from driada.recurrence.rqa import compute_rqa
from driada.network import Network

# ---------------------------------------------------------------------------
# Population config (same as recurrence_population example)
# ---------------------------------------------------------------------------
POPULATION = [
    {"name": "event_0",   "count": 30, "features": ["event_0"]},
    {"name": "event_1",   "count": 30, "features": ["event_1"]},
    {"name": "event_2",   "count": 30, "features": ["event_2"]},
    {"name": "event_0|1", "count": 10, "features": ["event_0", "event_1"],
     "combination": "or"},
    {"name": "event_0|2", "count": 10, "features": ["event_0", "event_2"],
     "combination": "or"},
    {"name": "event_1|2", "count": 10, "features": ["event_1", "event_2"],
     "combination": "or"},
]

MODULE_COLORS = {
    "event_0":   "#1a5acd",
    "event_1":   "#ffaa00",
    "event_2":   "#33cc33",
    "event_0|1": "#cc44cc",
    "event_0|2": "#00dddd",
    "event_1|2": "#ff4444",
}
MODULE_SHORT = {
    "event_0": "E0", "event_1": "E1", "event_2": "E2",
    "event_0|1": "E0|E1", "event_0|2": "E0|E2",
    "event_1|2": "E1|E2",
}

OUT_DIR = Path(__file__).parent / 'results' / 'synthetic'


def get_neuron_modules():
    """Map neuron index -> module name."""
    modules = {}
    idx = 0
    for group in POPULATION:
        for _ in range(group["count"]):
            modules[idx] = group["name"]
            idx += 1
    return modules


def parse_args():
    p = argparse.ArgumentParser(description='Synthetic recurrence analysis')
    p.add_argument('--duration', type=int, default=600,
                   help='Recording duration in seconds (default 600)')
    p.add_argument('--fps', type=float, default=5,
                   help='Sampling rate in Hz (default 5)')
    p.add_argument('--k', type=int, default=50, help='k-NN for recurrence graph')
    p.add_argument('--jrp-threshold', type=float, default=None,
                   help='Binarization threshold (auto if None)')
    p.add_argument('--n-clusters', type=int, default=4,
                   help='Number of SpectralClustering clusters (default 4: '
                        '3 events + baseline)')
    p.add_argument('--n-jobs', type=int, default=-1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-cache', action='store_true',
                   help='Force recompute (ignore cached mean matrix)')
    p.add_argument('--tau-method', default='first_minimum',
                   choices=['first_minimum', 'exponential_fit'])
    p.add_argument('--max-dim', type=int, default=10)
    return p.parse_args()


def savefig(fig, name, out_dir):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    neuron_modules = get_neuron_modules()
    module_names = sorted(set(neuron_modules.values()))
    n_total = sum(g["count"] for g in POPULATION)

    # ==================================================================
    # Step 1. Generate synthetic experiment
    # ==================================================================
    print(f'\n{"="*60}')
    print(f'  Synthetic Population Recurrence Analysis')
    print(f'{"="*60}')

    exp = generate_tuned_selectivity_exp(
        population=POPULATION,
        duration=args.duration,
        fps=args.fps,
        seed=args.seed,
        n_discrete_features=3,
        baseline_rate=0.05,
        peak_rate=2.0,
        decay_time=2.0,
        calcium_noise=0.02,
        verbose=False,
    )

    calcium = exp.calcium.data
    n_neurons, n_frames = calcium.shape
    fps = args.fps
    print(f'  Neurons: {n_neurons}, Frames: {n_frames} ({fps:.0f} fps, '
          f'{n_frames/fps:.0f}s)')
    for g in POPULATION:
        print(f'    {g["count"]:>2} {g["name"]:<12}')

    # Extract event time series (ground truth)
    events = {}
    for i in range(3):
        key = f'event_{i}'
        if key in exp.dynamic_features:
            events[key] = exp.dynamic_features[key].data
    n_events = len(events)
    print(f'  Events: {n_events} ({", ".join(events.keys())})')

    # Composite event label: which event(s) are active at each time point
    # 0 = baseline, 1 = event_0, 2 = event_1, 3 = event_2, 4+ = overlap
    event_label = np.zeros(n_frames, dtype=int)  # 0 = baseline
    for i, (key, ev) in enumerate(events.items()):
        event_label[ev > 0.5] = i + 1
    # Count overlaps
    event_sum = sum((ev > 0.5).astype(int) for ev in events.values())
    event_label[event_sum > 1] = n_events + 1  # overlap

    for lbl in range(n_events + 2):
        cnt = (event_label == lbl).sum()
        if cnt > 0:
            names = {0: 'baseline', n_events + 1: 'overlap'}
            for i in range(n_events):
                names[i + 1] = f'event_{i}'
            print(f'    {names.get(lbl, "?"):>12}: {cnt} pts '
                  f'({100*cnt/n_frames:.1f}%)')

    # ==================================================================
    # Step 2. Build MultiTimeSeries -> population recurrence graph
    # ==================================================================
    mts = MultiTimeSeries(calcium, discrete=False)

    tm_tag = 'ef' if args.tau_method == 'exponential_fit' else 'fm'
    cache_file = OUT_DIR / f'mean_matrix_d{args.duration}_fps{args.fps:.0f}_k{args.k}_{tm_tag}_md{args.max_dim}.npz'

    if cache_file.exists() and not args.no_cache:
        print(f'\n--- Step 2: Loading cached mean matrix ---')
        cached = np.load(cache_file, allow_pickle=True)
        mean_matrix = cached['mean_matrix']
        taus = cached['taus']
        dims = cached['dims']
        rqa_dicts = cached['rqa_dicts'].item() if 'rqa_dicts' in cached else None
        min_n = mean_matrix.shape[0]
        print(f'  Loaded: {min_n} time points')
    else:
        print(f'\n--- Step 2: Population recurrence graph (mean, k={args.k}) ---')
        t2 = time.time()

        pop_rg = mts.population_recurrence_graph(
            method='mean', k=args.k, n_jobs=args.n_jobs,
            tau_method=args.tau_method, max_dim=args.max_dim)

        mean_matrix = pop_rg.adj.toarray() if sp.issparse(pop_rg.adj) else pop_rg.adj
        min_n = mean_matrix.shape[0]
        print(f'  Population graph: {min_n} time points, {time.time()-t2:.1f}s')

        # Extract per-neuron tau/dim/RQA
        print(f'  Extracting per-neuron stats...')
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
        print(f'  Cached: {cache_file}')

    median_tau = int(np.median(taus))
    median_dim = int(np.median(dims))
    measures = ['DET', 'LAM', 'ENTR', 'L_mean', 'L_max', 'TT']

    # Per-neuron stats by module
    print(f'\n--- Per-neuron stats by module ---')
    print(f'  {"Module":<12} {"n":>3} {"tau":>6} {"dim":>5} '
          f'{"DET":>7} {"LAM":>7} {"ENTR":>7}')
    print(f'  {"-"*52}')
    groups = {}
    for idx, m in neuron_modules.items():
        groups.setdefault(m, []).append(idx)
    for m in module_names:
        idxs = groups[m]
        if rqa_dicts is not None:
            det = rqa_dicts['DET'][idxs].mean()
            lam = rqa_dicts['LAM'][idxs].mean()
            entr = rqa_dicts['ENTR'][idxs].mean()
        else:
            det = lam = entr = float('nan')
        print(f'  {MODULE_SHORT.get(m,m):<12} {len(idxs):>3} '
              f'{taus[idxs].mean():>5.0f} {dims[idxs].mean():>5.1f} '
              f'{det:>6.3f} {lam:>6.3f} {entr:>6.3f}')

    # Fig 1: tau/dim distributions by module
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for m in module_names:
        idxs = groups[m]
        ax1.hist(taus[idxs], bins=15, alpha=0.5,
                 color=MODULE_COLORS[m], label=MODULE_SHORT[m])
        ax2.hist(dims[idxs], bins=range(1, 12), alpha=0.5,
                 color=MODULE_COLORS[m], label=MODULE_SHORT[m], align='left')
    ax1.set_xlabel('tau (samples)')
    ax1.set_ylabel('Count')
    ax1.legend(fontsize=8)
    ax2.set_xlabel('Embedding dim m')
    ax2.set_ylabel('Count')
    ax2.legend(fontsize=8)
    fig.suptitle(f'Synthetic: tau and dim by module (n={n_neurons})')
    fig.tight_layout()
    savefig(fig, 'fig01_tau_dim.png', OUT_DIR)

    # ==================================================================
    # Step 3. Population recurrence heatmap + event overlay
    # ==================================================================
    # Mask diagonal band
    diag_band = (np.abs(np.arange(min_n)[:, None] - np.arange(min_n)[None, :])
                 < median_tau * 3)
    mean_display = mean_matrix.copy()
    mean_display[diag_band] = 0

    vmax = (np.percentile(mean_display[mean_display > 0], 99)
            if np.any(mean_display > 0) else 1)

    # Align event labels to embedded length
    offset = n_frames - min_n
    event_label_aligned = event_label[offset:offset + min_n]
    time_aligned = np.arange(min_n) / fps

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), width_ratios=[1, 1])
    ax = axes[0]
    im = ax.imshow(mean_display, cmap='hot', aspect='equal', origin='lower',
                   vmin=0, vmax=vmax, interpolation='none')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Time index')
    ax.set_title(f'Population recurrence ({n_neurons} neurons, mean)')
    plt.colorbar(im, ax=ax, label='Fraction of recurring neurons', shrink=0.8)

    # Right: event labels over time
    ax = axes[1]
    event_colors = ['#cccccc', '#1a5acd', '#ffaa00', '#33cc33', '#ff4444']
    event_names = ['baseline', 'event_0', 'event_1', 'event_2', 'overlap']
    for lbl in range(min(n_events + 2, len(event_colors))):
        mask = event_label_aligned == lbl
        if mask.any():
            ax.scatter(time_aligned[mask], np.full(mask.sum(), lbl),
                       c=event_colors[lbl], s=2, alpha=0.6,
                       label=event_names[lbl])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Event')
    ax.set_yticks(range(len(event_names)))
    ax.set_yticklabels(event_names, fontsize=9)
    ax.set_title('Ground-truth events')
    ax.legend(markerscale=5, fontsize=8)
    fig.tight_layout()
    savefig(fig, 'fig02_population_recurrence.png', OUT_DIR)

    # ==================================================================
    # Step 4. Binarize -> SpectralClustering -> compare with events
    # ==================================================================
    # Auto-select threshold: target ~5% recurrence rate
    offdiag = mean_display[~diag_band]
    if args.jrp_threshold is not None:
        threshold = args.jrp_threshold
    else:
        # Target RR ~ 5%
        threshold = np.percentile(offdiag[offdiag > 0], 95) if np.any(offdiag > 0) else 0.01
        # Ensure minimum number of edges
        jrp_test = (mean_display >= threshold).astype(float)
        if jrp_test.sum() < 100:
            for thr in [0.01, 0.005, 0.002, 0.001]:
                jrp_test = (mean_display >= thr).astype(float)
                if jrp_test.sum() >= 100:
                    threshold = thr
                    break

    print(f'\n--- Step 4: JRP clustering (threshold={threshold:.4f}) ---')
    jrp_binary = (mean_display >= threshold).astype(float)
    jrp_sparse = sp.csr_matrix(jrp_binary)
    nnz = int(jrp_binary.sum())
    n_offdiag = min_n * (min_n - 1)
    rr = nnz / n_offdiag if n_offdiag > 0 else 0
    print(f'  nnz={nnz}, RR={rr:.4f}')

    jrp_rqa = compute_rqa(jrp_sparse)
    print(f'  DET={jrp_rqa["DET"]:.3f}, LAM={jrp_rqa["LAM"]:.3f}, '
          f'ENTR={jrp_rqa["ENTR"]:.3f}')

    # Spectral clustering
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    n_cl = args.n_clusters
    if nnz > 50:
        sc = SpectralClustering(n_clusters=n_cl, affinity='precomputed',
                                random_state=args.seed, n_init=20)
        labels = sc.fit_predict(jrp_binary)
    else:
        print('  WARNING: JRP too sparse, random labels')
        labels = rng.integers(0, n_cl, size=min_n)

    # Compare with ground-truth event labels
    ari = adjusted_rand_score(event_label_aligned, labels)
    nmi = normalized_mutual_info_score(event_label_aligned, labels)
    print(f'\n  Clusters vs ground-truth events:')
    print(f'    ARI = {ari:.3f}')
    print(f'    NMI = {nmi:.3f}')

    # Cluster-event breakdown
    print(f'\n  {"Cluster":>8} {"n":>5}', end='')
    for en in event_names[:n_events+1]:
        print(f' {en:>10}', end='')
    print()
    for cl in range(n_cl):
        mask_cl = labels == cl
        print(f'  {cl:>8} {mask_cl.sum():>5}', end='')
        for lbl in range(n_events + 1):
            cnt = ((event_label_aligned == lbl) & mask_cl).sum()
            pct = 100 * cnt / mask_cl.sum() if mask_cl.sum() > 0 else 0
            print(f' {cnt:>5}({pct:>3.0f}%)', end='')
        print()

    # ==================================================================
    # Fig 3: Clusters vs events (main result)
    # ==================================================================
    cl_colors = plt.cm.Set1(np.linspace(0, 1, n_cl))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Top: event labels
    ax = axes[0]
    for lbl in range(n_events + 1):
        mask = event_label_aligned == lbl
        if mask.any():
            ax.scatter(time_aligned[mask],
                       np.full(mask.sum(), lbl),
                       c=event_colors[lbl], s=3, alpha=0.6,
                       label=event_names[lbl])
    ax.set_ylabel('Event')
    ax.set_title(f'Ground-truth events vs JRP clusters (ARI={ari:.3f}, '
                 f'NMI={nmi:.3f})')
    ax.legend(markerscale=5, fontsize=8)

    # Middle: cluster labels
    ax = axes[1]
    for cl in range(n_cl):
        mask = labels == cl
        ax.scatter(time_aligned[mask], np.full(mask.sum(), cl),
                   c=[cl_colors[cl]], s=3, alpha=0.6,
                   label=f'Cluster {cl}')
    ax.set_ylabel('Cluster')
    ax.legend(markerscale=5, fontsize=8)

    # Bottom: event density (rolling fraction of active events)
    ax = axes[2]
    win = int(10 * fps)  # 10-second window
    for i in range(n_events):
        key = f'event_{i}'
        ev = events[key][offset:offset + min_n]
        density = np.convolve(ev > 0.5, np.ones(win)/win, mode='same')
        ax.plot(time_aligned, density, color=event_colors[i+1],
                alpha=0.7, label=event_names[i+1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Event density')
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, 'fig03_clusters_vs_events.png', OUT_DIR)

    # Fig 4: Confusion matrix (clusters x events)
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    # Build confusion matrix
    n_event_labels = n_events + 1  # baseline + 3 events
    conf = np.zeros((n_cl, n_event_labels), dtype=int)
    for cl in range(n_cl):
        for lbl in range(n_event_labels):
            conf[cl, lbl] = ((labels == cl) & (event_label_aligned == lbl)).sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf, cmap='Blues', aspect='auto')
    for i in range(n_cl):
        for j in range(n_event_labels):
            val = conf[i, j]
            if val > 0:
                color = 'white' if val > conf.max() / 2 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
    ax.set_xticks(range(n_event_labels))
    ax.set_xticklabels(event_names[:n_event_labels], fontsize=9)
    ax.set_yticks(range(n_cl))
    ax.set_yticklabels([f'Cluster {i}' for i in range(n_cl)], fontsize=9)
    ax.set_xlabel('Ground-truth event')
    ax.set_ylabel('JRP cluster')
    ax.set_title(f'Confusion: JRP clusters vs events (ARI={ari:.3f})')
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    savefig(fig, 'fig04_confusion_matrix.png', OUT_DIR)

    # ==================================================================
    # Step 5. Windowed RQA vs event density
    # ==================================================================
    print(f'\n--- Step 5: Windowed RQA ---')
    win_size = int(25 * fps)  # ~25 sec
    step = win_size // 2
    n_windows = (min_n - win_size) // step + 1

    win_det = np.full(n_windows, np.nan)
    win_lam = np.full(n_windows, np.nan)
    win_event_frac = np.full(n_windows, np.nan)
    win_time = np.full(n_windows, np.nan)

    any_event = (event_label_aligned > 0).astype(float)

    for wi in range(n_windows):
        i0 = wi * step
        i1 = i0 + win_size
        win_adj = jrp_sparse[i0:i1, i0:i1]
        if win_adj.nnz > 10:
            rqa_w = compute_rqa(win_adj)
            win_det[wi] = rqa_w['DET']
            win_lam[wi] = rqa_w['LAM']
        win_event_frac[wi] = any_event[i0:i1].mean()
        win_time[wi] = time_aligned[i0 + win_size // 2]

    valid = ~np.isnan(win_det) & ~np.isnan(win_event_frac)
    if valid.sum() > 3:
        r_det, p_det = stats.pearsonr(win_event_frac[valid], win_det[valid])
        r_lam, p_lam = stats.pearsonr(win_event_frac[valid], win_lam[valid])
        print(f'  DET vs event fraction: r={r_det:.3f}, p={p_det:.3e}')
        print(f'  LAM vs event fraction: r={r_lam:.3f}, p={p_lam:.3e}')
    else:
        r_det = r_lam = np.nan

    # Fig 5: Windowed DET/LAM vs event density
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(win_time, win_event_frac, 'k-', alpha=0.7,
                 label='Event fraction')
    axes[0].set_ylabel('Event fraction')
    axes[0].set_title('Windowed analysis (synthetic)')
    axes[0].legend()
    lbl_d = f'DET (r={r_det:.2f})' if not np.isnan(r_det) else 'DET'
    lbl_l = f'LAM (r={r_lam:.2f})' if not np.isnan(r_lam) else 'LAM'
    axes[1].plot(win_time, win_det, 'b-', alpha=0.7, label=lbl_d)
    axes[1].plot(win_time, win_lam, 'r-', alpha=0.7, label=lbl_l)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('RQA')
    axes[1].legend()
    fig.tight_layout()
    savefig(fig, 'fig05_windowed_rqa_vs_events.png', OUT_DIR)

    # ==================================================================
    # Step 6. Jaccard network -> Louvain -> compare with modules
    # ==================================================================
    has_per_neuron = (hasattr(mts.ts_list[0], '_recurrence_graph_cache') and
                      mts.ts_list[0]._recurrence_graph_cache is not None)

    if has_per_neuron:
        print(f'\n--- Step 6: Jaccard network (neuron-neuron) ---')
        t6 = time.time()
        per_neuron_graphs = [ts._recurrence_graph_cache[1] for ts in mts.ts_list]
        sim_matrix, jac_mask = pairwise_jaccard_sparse(per_neuron_graphs)
        # jac_mask: boolean array, True for neurons kept after adaptive trim
        kept_indices = np.where(jac_mask)[0] if jac_mask is not None else np.arange(n_neurons)
        n_jac = sim_matrix.shape[0]
        upper = sim_matrix[np.triu_indices(n_jac, k=1)]
        print(f'  Jaccard: {n_jac} neurons (of {n_neurons}), '
              f'mean={upper.mean():.4f}, std={upper.std():.4f}')

        # Map Jaccard matrix indices back to original neuron indices
        # kept_indices[i] = original neuron index for Jaccard row/col i
        jac_neuron_modules = {i: neuron_modules[kept_indices[i]]
                              for i in range(n_jac)}

        # Within vs between module
        within_vals = []
        between_vals = []
        for i in range(n_jac):
            for j in range(i + 1, n_jac):
                if jac_neuron_modules[i] == jac_neuron_modules[j]:
                    within_vals.append(sim_matrix[i, j])
                else:
                    between_vals.append(sim_matrix[i, j])
        within_vals = np.array(within_vals)
        between_vals = np.array(between_vals)
        ratio = (within_vals.mean() / between_vals.mean()
                 if between_vals.mean() > 0 else float('inf'))
        print(f'  Within/between ratio: {ratio:.2f}x')

        # Louvain communities
        import networkx.algorithms.community as nx_comm

        thr_95 = np.percentile(upper, 90)
        sim_thr = sim_matrix.copy()
        sim_thr[sim_thr < thr_95] = 0
        np.fill_diagonal(sim_thr, 0)
        net = Network(adj=sp.csr_matrix(sim_thr), preprocessing='giant_cc',
                      create_nx_graph=True)

        communities = nx_comm.louvain_communities(net.graph, weight='weight',
                                                  seed=args.seed)
        communities = sorted(communities, key=len, reverse=True)

        # ARI: detected communities vs ground-truth modules
        # Note: community node IDs are Jaccard matrix indices (0..n_jac-1)
        nodes_in_net = set()
        for comm in communities:
            nodes_in_net.update(comm)

        true_labels_net = []
        detected_labels_net = []
        for ci, comm in enumerate(communities):
            for node in comm:
                detected_labels_net.append(ci)
                true_labels_net.append(jac_neuron_modules.get(node, 'unknown'))

        mod_to_int = {m: i for i, m in enumerate(module_names)}
        true_int = [mod_to_int.get(t, -1) for t in true_labels_net]
        ari_modules = adjusted_rand_score(true_int, detected_labels_net)

        print(f'  Communities: {len(communities)} '
              f'(sizes: {", ".join(str(len(c)) for c in communities)})')
        print(f'  ARI (modules): {ari_modules:.3f}')
        print(f'  Time: {time.time()-t6:.1f}s')

        # Fig 6: Jaccard matrix ordered by module
        # Build groups in Jaccard index space
        jac_groups = {}
        for ji in range(n_jac):
            m = jac_neuron_modules[ji]
            jac_groups.setdefault(m, []).append(ji)

        mod_order = []
        for m in module_names:
            if m in jac_groups:
                mod_order.extend(sorted(jac_groups[m]))
        mod_arr = np.array(mod_order)
        sim_by_mod = sim_matrix[np.ix_(mod_arr, mod_arr)]
        np.fill_diagonal(sim_by_mod, 0)

        vmin_j = np.percentile(upper, 5)
        vmax_j = np.percentile(upper, 99)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        im1 = axes[0].imshow(sim_by_mod, cmap='inferno', aspect='auto',
                             vmin=vmin_j, vmax=vmax_j)
        cumsum_mod = np.cumsum([0] + [len(jac_groups.get(m, []))
                                       for m in module_names])
        for b in cumsum_mod[1:-1]:
            axes[0].axhline(b - 0.5, color='white', lw=0.8, alpha=0.7)
            axes[0].axvline(b - 0.5, color='white', lw=0.8, alpha=0.7)
        axes[0].set_title(f'Jaccard (by module, ratio={ratio:.2f}x)')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        # Ordered by detected community
        comm_order = []
        for comm in communities:
            comm_order.extend(sorted(comm))
        missing = [i for i in range(n_jac) if i not in nodes_in_net]
        comm_order.extend(missing)
        comm_arr = np.array(comm_order)
        sim_by_comm = sim_matrix[np.ix_(comm_arr, comm_arr)]
        np.fill_diagonal(sim_by_comm, 0)

        im2 = axes[1].imshow(sim_by_comm, cmap='inferno', aspect='auto',
                             vmin=vmin_j, vmax=vmax_j)
        cumsum_comm = np.cumsum(
            [0] + [len(c) for c in communities] + [len(missing)])
        for b in cumsum_comm[1:-1]:
            axes[1].axhline(b - 0.5, color='white', lw=0.8, alpha=0.7)
            axes[1].axvline(b - 0.5, color='white', lw=0.8, alpha=0.7)
        axes[1].set_title(f'Jaccard (by community, ARI={ari_modules:.3f})')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        fig.suptitle('Neuron-neuron Jaccard similarity')
        fig.tight_layout()
        savefig(fig, 'fig06_jaccard_similarity.png', OUT_DIR)

        # Fig 6b: Confusion matrix (modules vs communities)
        n_comm = len(communities)
        conf_mod = np.zeros((len(module_names), n_comm), dtype=int)
        for ci, comm in enumerate(communities):
            for node in comm:
                m = jac_neuron_modules.get(node, 'unknown')
                if m in module_names:
                    mi = module_names.index(m)
                    conf_mod[mi, ci] += 1

        fig, ax = plt.subplots(figsize=(max(6, n_comm * 0.8 + 2),
                                        max(5, len(module_names) * 0.7 + 1)))
        im = ax.imshow(conf_mod, cmap='Blues', aspect='auto')
        for i in range(len(module_names)):
            for j in range(n_comm):
                val = conf_mod[i, j]
                if val > 0:
                    color = 'white' if val > conf_mod.max() / 2 else 'black'
                    ax.text(j, i, str(val), ha='center', va='center',
                            fontsize=10, color=color, fontweight='bold')
        ax.set_xticks(range(n_comm))
        ax.set_xticklabels([f'C{i}' for i in range(n_comm)], fontsize=9)
        ax.set_yticks(range(len(module_names)))
        ax.set_yticklabels([MODULE_SHORT.get(m, m) for m in module_names],
                           fontsize=9)
        ax.set_xlabel('Detected community')
        ax.set_ylabel('Ground-truth module')
        ax.set_title(f'Module recovery (ARI={ari_modules:.3f})')
        plt.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        savefig(fig, 'fig06b_module_confusion.png', OUT_DIR)
    else:
        print(f'\n--- Step 6: Skipped (loaded from cache, no per-neuron graphs) ---')
        ari_modules = np.nan
        ratio = np.nan

    # ==================================================================
    # Step 7. Shuffle control
    # ==================================================================
    print(f'\n--- Step 7: Shuffle control ---')
    t7 = time.time()

    # Circular shift shuffle (same as NOF script)
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

    real_offdiag = mean_display[:min_n_s, :min_n_s][~diag_band_s]
    shuf_offdiag = mean_shuf[~diag_band_s]
    print(f'  Real mean recurrence:     {real_offdiag.mean():.5f}')
    print(f'  Shuffled mean recurrence: {shuf_offdiag.mean():.5f}')

    # RQA comparison
    jrp_shuf_binary = (mean_shuf >= threshold).astype(float)
    jrp_shuf_sparse = sp.csr_matrix(jrp_shuf_binary)
    jrp_shuf_rqa = compute_rqa(jrp_shuf_sparse)

    jrp_real_trimmed = jrp_sparse[:min_n_s, :min_n_s]
    jrp_real_rqa = compute_rqa(jrp_real_trimmed)

    print(f'\n  RQA comparison:')
    print(f'  {"":12s} {"Real":>8s} {"Shuffled":>8s} {"Ratio":>8s}')
    print(f'  {"-"*40}')
    for m in ['DET', 'LAM', 'ENTR']:
        rv = jrp_real_rqa[m]
        sv = jrp_shuf_rqa[m]
        ratio_rqa = rv / sv if sv > 0 else float('inf')
        print(f'  {m:12s} {rv:8.3f} {sv:8.3f} {ratio_rqa:8.2f}')

    print(f'  Time: {time.time()-t7:.1f}s')

    # Fig 7: Real vs shuffled
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
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
    fig.suptitle('Population recurrence: real vs shuffled (synthetic)')
    fig.tight_layout()
    savefig(fig, 'fig07_real_vs_shuffled.png', OUT_DIR)

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
    ax.set_title('Population JRP structure: real vs shuffled (synthetic)')
    ax.legend()
    savefig(fig, 'fig08_rqa_real_vs_shuffled.png', OUT_DIR)

    # ==================================================================
    # Summary
    # ==================================================================
    total = time.time() - t0
    print(f'\n{"="*60}')
    print(f'  SUMMARY')
    print(f'{"="*60}')
    print(f'  Population: {n_neurons} neurons, {len(POPULATION)} modules')
    print(f'  Duration: {args.duration}s @ {fps:.0f} fps')
    print(f'  Recurrence: k={args.k}, threshold={threshold:.4f}')
    print(f'')
    print(f'  JRP clusters vs events:')
    print(f'    ARI  = {ari:.3f}')
    print(f'    NMI  = {nmi:.3f}')
    if not np.isnan(r_det):
        print(f'    DET-event r = {r_det:.3f}')
    print(f'')
    if not np.isnan(ari_modules):
        print(f'  Jaccard communities vs modules:')
        print(f'    ARI  = {ari_modules:.3f}')
        print(f'    Within/between ratio = {ratio:.2f}x')
    print(f'')
    print(f'  Real vs shuffled RQA:')
    for m in ['DET', 'LAM']:
        rv = jrp_real_rqa[m]
        sv = jrp_shuf_rqa[m]
        print(f'    {m}: {rv:.3f} vs {sv:.3f} '
              f'({rv/sv:.1f}x)' if sv > 0 else f'    {m}: {rv:.3f} vs 0')
    print(f'')
    print(f'  Done in {total:.1f}s ({total/60:.1f} min)')
    print(f'  Results: {OUT_DIR}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
