#!/usr/bin/env python
"""
Test 2: cluster time windows by their RQA vector (not individual time points).

Hypothesis: collective modes = clusters of windows in RQA space.
Speed-of-stops should fall into a "high DET/LAM" cluster.

Pipeline per session:
  1. Load cached mean_matrix (JRP) + binarize at session threshold
  2. Slide 25-s windows (50% overlap), compute RQA vector for each
  3. KMeans / GMM clustering on window RQA vectors (k=2)
  4. Test: median speed difference between clusters (Mann-Whitney U)
  5. Plot: RQA vectors colored by cluster + cluster speed distributions

Run on multiple sessions for cohort-level test.

Usage:
    python cluster_windows_by_rqa.py --session NOF_H01_1D
    python cluster_windows_by_rqa.py --batch
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DRIADA_ROOT = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA_ROOT / 'src'))
sys.path.insert(0, str(DRIADA_ROOT / 'tools'))

from load_synchronized_experiments import load_experiment_from_npz
from driada.recurrence.rqa import compute_rqa

NOF_DIR = DRIADA_ROOT / 'DRIADA data' / 'NOF' / 'SynchronizedData26_v1'
LNOF_DIR = DRIADA_ROOT / 'DRIADA data' / 'LNOF' / 'aligned'
RESULTS = Path(__file__).parent / 'results'


def npz_path(session):
    if session.startswith('LNOF'):
        return LNOF_DIR / f'{session}_aligned.npz'
    return NOF_DIR / f'{session}_aligned.npz'


def find_cache(session, ds=5, k=50):
    """Find existing mean_matrix cache for the session."""
    out = RESULTS / session
    for pat in [f'mean_matrix_ds{ds}_k{k}_exp_md3.npz',
                f'mean_matrix_ds{ds}_k{k}_exp_md2.npz',
                f'mean_matrix_ds{ds}_k20_exp_md3.npz']:
        path = out / pat
        if path.exists():
            return path
    return None


def cluster_session(session, ds=5, k=50, win_sec=25.0, overlap=0.5,
                    threshold=None, n_clusters=2, smooth_sigma=2.0, seed=42,
                    use_shuffled=False):
    """Run window-RQA clustering on one session. Returns dict of results.

    use_shuffled: if True, analyse shuffled mean matrix. Threshold is still
    taken from the real cache (same recurrence-rate floor for fair comparison).
    """
    out = RESULTS / session
    cache = find_cache(session, ds=ds, k=k)
    if cache is None:
        print(f'  {session}: no cache found, skipping')
        return None

    print(f'  Loading: {cache.name}')
    cached = np.load(cache, allow_pickle=True)
    real_mean = cached['mean_matrix']
    taus = cached['taus']
    median_tau = int(np.median(taus))

    if use_shuffled:
        shuf_path = out / cache.name.replace('mean_matrix_', 'mean_matrix_shuffled_')
        if not shuf_path.exists():
            print(f'  {session}: no shuffled cache ({shuf_path.name}), skipping')
            return None
        print(f'  Loading shuffled: {shuf_path.name}')
        shuf_mean = np.load(shuf_path, allow_pickle=True)['mean_matrix']
        # crop both to common size
        n = min(real_mean.shape[0], shuf_mean.shape[0])
        real_mean = real_mean[:n, :n]
        mean_matrix = shuf_mean[:n, :n]
    else:
        mean_matrix = real_mean

    min_n = mean_matrix.shape[0]

    # Load speed
    exp = load_experiment_from_npz(npz_path(session), verbose=False)
    speed_ds = exp.dynamic_features['speed'].data[::ds]
    n_frames_full = exp.calcium.data.shape[1] // ds + (1 if exp.calcium.data.shape[1] % ds else 0)
    offset = n_frames_full - min_n
    speed_aligned = speed_ds[offset:offset + min_n]
    if len(speed_aligned) < min_n:
        speed_aligned = np.pad(speed_aligned, (0, min_n - len(speed_aligned)), mode='edge')
    fps_eff = 20.0 / ds

    # Mask diagonal, binarize
    diag = (np.abs(np.arange(min_n)[:, None] - np.arange(min_n)[None, :])
            < median_tau * 3)
    m = mean_matrix.copy()
    m[diag] = 0
    if threshold is None:
        # session-specific threshold: 95th percentile of off-diagonal
        # For shuffled: take threshold from REAL matrix (same RR floor)
        ref = real_mean.copy() if use_shuffled else m
        if use_shuffled:
            ref[diag] = 0
        threshold = np.percentile(ref[ref > 0], 95) if np.any(ref > 0) else 0.01
    jrp = (m >= threshold).astype(float)

    # Sliding windows
    win_size = int(win_sec * fps_eff)
    step = int(win_size * (1 - overlap))
    n_win = (min_n - win_size) // step + 1

    measures = ['DET', 'LAM', 'ENTR', 'TT', 'L_mean', 'L_max', 'DIV']
    rqa_vecs = np.full((n_win, len(measures)), np.nan)
    win_speed = np.full(n_win, np.nan)
    win_time = np.full(n_win, np.nan)

    for wi in range(n_win):
        i0 = wi * step
        i1 = i0 + win_size
        sub = sp.csr_matrix(jrp[i0:i1, i0:i1])
        if sub.nnz > 10:
            rqa = compute_rqa(sub)
            for mi, mname in enumerate(measures):
                rqa_vecs[wi, mi] = rqa.get(mname, np.nan)
        win_speed[wi] = speed_aligned[i0:i1].mean()
        win_time[wi] = (i0 + win_size // 2) / fps_eff

    # Drop windows with any nan
    valid = ~np.any(np.isnan(rqa_vecs), axis=1)
    rqa_v = rqa_vecs[valid]
    sp_v = win_speed[valid]
    tm_v = win_time[valid]

    if len(rqa_v) < 4:
        print(f'  {session}: only {len(rqa_v)} valid windows, skipping')
        return None

    # Standardize + KMeans
    X = StandardScaler().fit_transform(rqa_v)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    labels = km.fit_predict(X)

    # Sort clusters by mean DET (higher DET = more deterministic = stops)
    det_means = [rqa_v[labels == c, 0].mean() for c in range(n_clusters)]
    order = np.argsort(det_means)[::-1]   # high DET first
    relabel = {old: new for new, old in enumerate(order)}
    labels = np.array([relabel[l] for l in labels])

    # Speed comparison (sorted: cluster 0 = high DET)
    groups = [sp_v[labels == c] for c in range(n_clusters)]
    if all(len(g) > 1 for g in groups):
        if n_clusters == 2:
            U, p = stats.mannwhitneyu(groups[0], groups[1], alternative='less')
            test_name = 'Mann-Whitney U (high-DET < low-DET)'
        else:
            F, p = stats.f_oneway(*groups)
            U = F
            test_name = 'ANOVA F'
    else:
        U, p = np.nan, np.nan
        test_name = 'n/a'

    # eta^2
    ss_b = sum(len(g) * (g.mean() - sp_v.mean())**2 for g in groups)
    ss_t = ((sp_v - sp_v.mean())**2).sum()
    eta2 = ss_b / ss_t if ss_t > 0 else 0

    speed_means = [g.mean() for g in groups]
    speed_med = [np.median(g) for g in groups]

    result = {
        'session': session,
        'n_windows': len(rqa_v),
        'threshold': threshold,
        'rqa_vecs': rqa_v,
        'labels': labels,
        'win_speed': sp_v,
        'win_time': tm_v,
        'measures': measures,
        'test_stat': U,
        'p_value': p,
        'test_name': test_name,
        'eta2': eta2,
        'speed_means': speed_means,
        'speed_medians': speed_med,
        'cluster_sizes': [int((labels == c).sum()) for c in range(n_clusters)],
    }

    print(f'  {session}: n={len(rqa_v)} windows, sizes={result["cluster_sizes"]}')
    print(f'    Speed mean: C0(high-DET)={speed_means[0]:.2f}, '
          f'C{n_clusters-1}(low-DET)={speed_means[-1]:.2f}')
    print(f'    {test_name}: stat={U:.2f}, p={p:.3e}, eta2={eta2:.3f}')

    return result


def plot_session(result, out_dir):
    """Plot RQA vectors colored by cluster + speed distributions."""
    out_dir = out_dir / 'per_mouse'
    out_dir.mkdir(parents=True, exist_ok=True)
    rqa_v = result['rqa_vecs']
    labels = result['labels']
    sp_v = result['win_speed']
    tm_v = result['win_time']
    measures = result['measures']
    n_cl = labels.max() + 1
    colors = plt.cm.Set1(np.linspace(0, 1, n_cl))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: RQA-vector PC1 vs PC2 colored by cluster
    from sklearn.decomposition import PCA
    Xs = StandardScaler().fit_transform(rqa_v)
    pc = PCA(n_components=2).fit_transform(Xs)
    ax = axes[0, 0]
    for c in range(n_cl):
        m = labels == c
        ax.scatter(pc[m, 0], pc[m, 1], c=[colors[c]], s=40, alpha=0.7,
                   edgecolor='k', linewidth=0.3,
                   label=f'C{c} (n={m.sum()})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Windows in RQA-vector PCA space')
    ax.legend()

    # Panel B: speed distributions by cluster
    ax = axes[0, 1]
    for c in range(n_cl):
        m = labels == c
        ax.hist(sp_v[m], bins=15, alpha=0.5, color=colors[c],
                label=f'C{c}: median={np.median(sp_v[m]):.2f}',
                density=True)
    ax.set_xlabel('Window mean speed')
    ax.set_ylabel('Density')
    ax.set_title(f'Speed by cluster ({result["test_name"]}, p={result["p_value"]:.2e})')
    ax.legend()

    # Panel C: RQA measures by cluster (bar)
    ax = axes[1, 0]
    width = 0.8 / n_cl
    x = np.arange(len(measures))
    for c in range(n_cl):
        means = [rqa_v[labels == c, mi].mean() for mi in range(len(measures))]
        sems = [rqa_v[labels == c, mi].std() / np.sqrt((labels == c).sum())
                for mi in range(len(measures))]
        ax.bar(x + c * width - 0.4 + width/2, means, width,
               yerr=sems, label=f'C{c}', color=colors[c], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(measures, rotation=30)
    ax.set_ylabel('Value')
    ax.set_title('RQA measures by cluster (mean ± SEM)')
    ax.legend()

    # Panel D: timeline — cluster vs speed
    ax2 = axes[1, 1]
    ax2.plot(tm_v, sp_v, 'k-', alpha=0.5, lw=1)
    for c in range(n_cl):
        m = labels == c
        ax2.scatter(tm_v[m], sp_v[m], c=[colors[c]], s=40,
                    edgecolor='k', linewidth=0.3, label=f'C{c}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Window mean speed')
    ax2.set_title('Timeline: cluster assignment vs speed')
    ax2.legend()

    fig.suptitle(f'{result["session"]}: window-RQA clustering (n_win={result["n_windows"]})',
                 fontsize=13)
    fig.tight_layout()
    name = f'window_rqa_clustering_{result["session"]}.png'
    fig.savefig(out_dir / name, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_dir / name}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--session', default='NOF_H01_1D')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--shuffled', action='store_true',
                   help='Run on shuffled data (threshold inherited from real)')
    p.add_argument('--ds', type=int, default=5)
    p.add_argument('--k', type=int, default=50)
    p.add_argument('--win-sec', type=float, default=25.0)
    p.add_argument('--n-clusters', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def list_sessions():
    sessions = []
    for d in sorted(RESULTS.iterdir()):
        if d.is_dir() and (d.name.startswith('NOF') or d.name.startswith('LNOF')):
            sessions.append(d.name)
    return sessions


def main():
    args = parse_args()
    suffix = '_shuffled' if args.shuffled else ''
    out_dir = RESULTS / f'window_rqa_clustering{suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = list_sessions() if args.batch else [args.session]
    label = 'SHUFFLED' if args.shuffled else 'REAL'
    print(f'Window-RQA clustering [{label}]: {len(sessions)} sessions')
    print('=' * 60)

    cohort = []
    for s in sessions:
        print(f'\n[{s}]')
        t0 = time.time()
        r = cluster_session(s, ds=args.ds, k=args.k, win_sec=args.win_sec,
                            n_clusters=args.n_clusters, seed=args.seed,
                            use_shuffled=args.shuffled)
        if r is None:
            continue
        plot_session(r, out_dir)
        cohort.append(r)
        print(f'  Time: {time.time()-t0:.1f}s')

    if len(cohort) >= 2:
        # Cohort summary
        print('\n' + '=' * 60)
        print('COHORT SUMMARY')
        print('=' * 60)
        ps = [r['p_value'] for r in cohort]
        diffs = [r['speed_medians'][0] - r['speed_medians'][-1] for r in cohort]
        eta2s = [r['eta2'] for r in cohort]
        print(f'{"Session":<20} {"diff(slow-fast)":>16} {"eta2":>8} {"p":>10}')
        for r, d, e, p in zip(cohort, diffs, eta2s, ps):
            sig = '*' if p < 0.05 else ' '
            print(f'{r["session"]:<20} {d:>16.3f} {e:>8.3f} {p:>10.3e}{sig}')
        print(f'\nSessions with high-DET cluster slower (sign test):')
        n_neg = sum(1 for d in diffs if d < 0)
        n_total = len(diffs)
        binom_p = stats.binomtest(n_neg, n_total, p=0.5,
                                  alternative='greater').pvalue
        print(f'  {n_neg}/{n_total} sessions, binomial p={binom_p:.3e}')

        # Save cohort summary
        data_dir = out_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(data_dir / 'cohort_summary.npz',
                 sessions=[r['session'] for r in cohort],
                 p_values=ps, diffs=diffs, eta2s=eta2s,
                 binom_p=binom_p)
        print(f'\n  Saved: {data_dir}/cohort_summary.npz')


if __name__ == '__main__':
    main()
