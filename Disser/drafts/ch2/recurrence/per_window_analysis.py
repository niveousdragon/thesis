#!/usr/bin/env python
"""
Per-window RQA analysis across NOF Day-1 cohort:
  1. Compute RQA vectors per window for each session (real + shuffled)
  2. Aggregate cohort PCA (color by speed / by mouse / by cluster)
  3. Per-metric correlation with speed (table)

Output:
  results/window_rqa_clustering/
    per_window_<session>.npz            (rqa_vecs, speeds, times)
    fig_cohort_pca.png                  (3 panels: speed/mouse/clust)
    fig_per_metric_speed.png            (per-RQA-metric correlation with speed)
    per_metric_speed.csv                (r and p per metric per session)
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
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DRIADA_ROOT = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA_ROOT / 'src'))
sys.path.insert(0, str(DRIADA_ROOT / 'tools'))
from load_synchronized_experiments import load_experiment_from_npz
from driada.recurrence.rqa import compute_rqa

NOF_DIR = DRIADA_ROOT / 'DRIADA data' / 'NOF' / 'SynchronizedData26_v1'
RESULTS = Path(__file__).parent / 'results'
OUT = RESULTS / 'window_rqa_clustering'
DATA = OUT / 'data'
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

DS = 5
K = 50
WIN_SEC = 25.0
OVERLAP = 0.5
MEASURES = ['DET', 'LAM', 'ENTR', 'TT', 'L_mean', 'L_max', 'DIV']


def list_nof_1d(days='1D'):
    pat = 'NOF_*_aligned.npz' if days == 'all' else f'NOF_*_{days}_aligned.npz'
    return sorted([p.stem.replace('_aligned', '') for p in NOF_DIR.glob(pat)])


def compute_windows(session, shuffled=False):
    """Compute per-window RQA + speed for one session."""
    out = RESULTS / session
    cache = out / f'mean_matrix_ds{DS}_k{K}_exp_md3.npz'
    if not cache.exists():
        return None
    cached = np.load(cache, allow_pickle=True)
    real_mean = cached['mean_matrix']
    taus = cached['taus']
    median_tau = int(np.median(taus))

    if shuffled:
        sp_path = out / f'mean_matrix_shuffled_ds{DS}_k{K}_exp_md3.npz'
        if not sp_path.exists():
            return None
        sm = np.load(sp_path, allow_pickle=True)['mean_matrix']
        n = min(real_mean.shape[0], sm.shape[0])
        real_mean = real_mean[:n, :n]
        mean_matrix = sm[:n, :n]
    else:
        mean_matrix = real_mean
    min_n = mean_matrix.shape[0]

    # Load speed
    npz = NOF_DIR / f'{session}_aligned.npz'
    exp = load_experiment_from_npz(npz, verbose=False)
    speed_ds = exp.dynamic_features['speed'].data[::DS]
    n_full = exp.calcium.data.shape[1] // DS + (1 if exp.calcium.data.shape[1] % DS else 0)
    offset = n_full - min_n
    speed = speed_ds[offset:offset + min_n]
    if len(speed) < min_n:
        speed = np.pad(speed, (0, min_n - len(speed)), mode='edge')
    fps_eff = 20.0 / DS

    # Mask diagonal, binarize at threshold from real
    diag = (np.abs(np.arange(min_n)[:, None] - np.arange(min_n)[None, :])
            < median_tau * 3)
    ref = real_mean.copy(); ref[diag] = 0
    threshold = np.percentile(ref[ref > 0], 95) if np.any(ref > 0) else 0.01
    m = mean_matrix.copy(); m[diag] = 0
    jrp = (m >= threshold).astype(float)

    # Sliding windows
    win_size = int(WIN_SEC * fps_eff)
    step = int(win_size * (1 - OVERLAP))
    n_win = (min_n - win_size) // step + 1
    rqa_vecs = np.full((n_win, len(MEASURES)), np.nan)
    speeds = np.full(n_win, np.nan)
    times = np.full(n_win, np.nan)
    for wi in range(n_win):
        i0 = wi * step; i1 = i0 + win_size
        sub = sp.csr_matrix(jrp[i0:i1, i0:i1])
        if sub.nnz > 10:
            r = compute_rqa(sub)
            for mi, mn in enumerate(MEASURES):
                rqa_vecs[wi, mi] = r.get(mn, np.nan)
        speeds[wi] = speed[i0:i1].mean()
        times[wi] = (i0 + win_size//2) / fps_eff

    valid = ~np.any(np.isnan(rqa_vecs), axis=1)
    return rqa_vecs[valid], speeds[valid], times[valid]


def gather_cohort(days='1D'):
    """Compute per-window data for all NOF sessions (filtered by days), cache to disk."""
    sessions = list_nof_1d(days)
    cohort = []   # list of dicts: session, real, shuf
    for s in sessions:
        cache = DATA / f'per_window_{s}.npz'
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            cohort.append({
                'session': s,
                'rqa_real': d['rqa_real'], 'speed_real': d['speed_real'],
                'rqa_shuf': d['rqa_shuf'], 'speed_shuf': d['speed_shuf'],
            })
            print(f'  {s}: loaded ({len(d["rqa_real"])} windows)')
            continue
        t0 = time.time()
        real = compute_windows(s, shuffled=False)
        shuf = compute_windows(s, shuffled=True)
        if real is None or shuf is None:
            print(f'  {s}: skipped (no cache)')
            continue
        rqa_r, sp_r, _ = real
        rqa_s, sp_s, _ = shuf
        np.savez_compressed(cache,
                            rqa_real=rqa_r, speed_real=sp_r,
                            rqa_shuf=rqa_s, speed_shuf=sp_s)
        cohort.append({'session': s,
                       'rqa_real': rqa_r, 'speed_real': sp_r,
                       'rqa_shuf': rqa_s, 'speed_shuf': sp_s})
        print(f'  {s}: {len(rqa_r)} windows ({time.time()-t0:.1f}s)')
    return cohort


def plot_cohort_pca(cohort):
    # concat across sessions; remember session index for per-mouse coloring
    Xs, ys, mids, Xshs = [], [], [], []
    for i, c in enumerate(cohort):
        Xs.append(c['rqa_real']); ys.append(c['speed_real'])
        mids.append(np.full(len(c['rqa_real']), i, dtype=int))
        Xshs.append(c['rqa_shuf'])
    X = np.vstack(Xs); y = np.concatenate(ys); mid = np.concatenate(mids)
    Xs2 = np.vstack(Xshs)
    n_sess = len(cohort)
    print(f'Cohort: {n_sess} sessions, {len(X)} real windows, {len(Xs2)} shuf')

    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=2).fit(scaler.transform(X))
    pc = pca.transform(scaler.transform(X))
    pcS = pca.transform(scaler.transform(Xs2))    # transform shuf via real fit
    print(f'  PC1 expl. var: {pca.explained_variance_ratio_[0]:.3f}, '
          f'PC2: {pca.explained_variance_ratio_[1]:.3f}')

    # cohort-wide clustering for panel 3 (k=2)
    km = KMeans(n_clusters=2, random_state=42, n_init=20)
    labels = km.fit_predict(scaler.transform(X))
    # ensure cluster 0 = high DET
    det_means = [X[labels == c, 0].mean() for c in (0, 1)]
    if det_means[1] > det_means[0]:
        labels = 1 - labels

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cm_speed = plt.cm.plasma
    vmin, vmax = np.percentile(y, [2, 98])
    sc = axes[0].scatter(pc[:, 0], pc[:, 1], c=y, cmap=cm_speed,
                         s=15, alpha=0.55, edgecolor='none',
                         vmin=vmin, vmax=vmax)
    axes[0].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
    axes[0].set_title(f'A. Real windows (n={len(X)}) — color = mean speed')
    plt.colorbar(sc, ax=axes[0], shrink=0.85, label='speed (cm/s)')

    cm_m = plt.cm.tab20
    sc2 = axes[1].scatter(pc[:, 0], pc[:, 1], c=mid, cmap=cm_m,
                          s=15, alpha=0.6, edgecolor='none',
                          vmin=0, vmax=n_sess)
    axes[1].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
    axes[1].set_title(f'B. Same windows colored by mouse (n={n_sess})')

    # Panel C: real vs shuffled in same PCA basis
    axes[2].scatter(pcS[:, 0], pcS[:, 1], c='gray', s=12, alpha=0.35,
                    edgecolor='none', label=f'Shuffled (n={len(pcS)})')
    axes[2].scatter(pc[labels == 1, 0], pc[labels == 1, 1], c='#888',
                    s=12, alpha=0.55, edgecolor='none',
                    label=f'Real, low-DET (n={(labels==1).sum()})')
    axes[2].scatter(pc[labels == 0, 0], pc[labels == 0, 1], c='#cc2222',
                    s=12, alpha=0.65, edgecolor='none',
                    label=f'Real, high-DET (n={(labels==0).sum()})')
    axes[2].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
    axes[2].set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
    axes[2].set_title('C. Real (clusters) vs Shuffled in real-PCA basis')
    axes[2].legend(fontsize=9, loc='best')

    # Loadings annotation
    load = pca.components_  # 2 × M
    txt = 'PC1 loadings: ' + ', '.join(f'{m}={v:+.2f}'
                                       for m, v in zip(MEASURES, load[0]))
    fig.text(0.5, -0.02, txt, ha='center', fontsize=9)

    fig.suptitle(
        f'Cohort PCA of RQA-vector windows (NOF Day-1, {n_sess} mice)',
        fontsize=13, y=1.02)
    fig.tight_layout()
    path = OUT / 'fig_cohort_pca.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def per_metric_speed(cohort):
    """For each metric, compute Pearson r with speed per session.
    Plot cohort distributions; save CSV.
    """
    rows = []   # (session, metric, r_real, p_real, r_shuf, p_shuf)
    for c in cohort:
        s = c['session']
        sp_r = c['speed_real']; sp_s = c['speed_shuf']
        for mi, m in enumerate(MEASURES):
            r_r, p_r = stats.pearsonr(c['rqa_real'][:, mi], sp_r)
            r_s, p_s = stats.pearsonr(c['rqa_shuf'][:, mi], sp_s)
            rows.append((s, m, r_r, p_r, r_s, p_s))

    csv = OUT / 'per_metric_speed.csv'
    with open(csv, 'w', encoding='utf-8') as f:
        f.write('session,metric,r_real,p_real,r_shuf,p_shuf\n')
        for row in rows:
            f.write('{},{},{:.4f},{:.4e},{:.4f},{:.4e}\n'.format(*row))
    print(f'Saved: {csv}')

    # Cohort plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    x = np.arange(len(MEASURES))
    width = 0.36
    for mi, m in enumerate(MEASURES):
        r_real = [row[2] for row in rows if row[1] == m]
        r_shuf = [row[4] for row in rows if row[1] == m]
        # box
        bp1 = ax.boxplot([r_real], positions=[mi - width/2], widths=width*0.85,
                         patch_artist=True, boxprops=dict(facecolor='#cc4444'),
                         medianprops=dict(color='black'),
                         flierprops=dict(marker='o', ms=3))
        bp2 = ax.boxplot([r_shuf], positions=[mi + width/2], widths=width*0.85,
                         patch_artist=True, boxprops=dict(facecolor='#bbbbbb'),
                         medianprops=dict(color='black'),
                         flierprops=dict(marker='o', ms=3))
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(MEASURES, rotation=20)
    ax.set_ylabel('Pearson r (metric vs window speed)')
    ax.set_title('A. Per-mouse r distribution per RQA metric '
                 '(red=real, gray=shuffled)')

    # Panel B: cohort median r per metric with paired Wilcoxon
    medians_r = []; medians_s = []; pvals = []; sign_neg = []
    for mi, m in enumerate(MEASURES):
        r_real = np.array([row[2] for row in rows if row[1] == m])
        r_shuf = np.array([row[4] for row in rows if row[1] == m])
        medians_r.append(np.median(r_real)); medians_s.append(np.median(r_shuf))
        try:
            w = stats.wilcoxon(r_real, alternative='less').pvalue
        except ValueError:
            w = 1.0
        pvals.append(w); sign_neg.append(int((r_real < 0).sum()))
    ax = axes[1]
    ax.bar(x - width/2, medians_r, width, color='#cc4444', label='Real')
    ax.bar(x + width/2, medians_s, width, color='#bbbbbb', label='Shuffled')
    for mi, (mr, p, ng) in enumerate(zip(medians_r, pvals, sign_neg)):
        ax.text(mi - width/2, mr - 0.02 if mr < 0 else mr + 0.01,
                f'{ng}/{len(cohort)}', ha='center', fontsize=8)
        if p < 0.001:
            ax.text(mi - width/2, mr - 0.04 if mr < 0 else mr + 0.04,
                    '***', ha='center', fontsize=10, color='red',
                    fontweight='bold')
        elif p < 0.01:
            ax.text(mi - width/2, mr - 0.04 if mr < 0 else mr + 0.04,
                    '**', ha='center', fontsize=10, color='red',
                    fontweight='bold')
        elif p < 0.05:
            ax.text(mi - width/2, mr - 0.04 if mr < 0 else mr + 0.04,
                    '*', ha='center', fontsize=10, color='red',
                    fontweight='bold')
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(MEASURES, rotation=20)
    ax.set_ylabel('Median Pearson r across mice')
    ax.set_title(f'B. Cohort median r per metric (Wilcoxon vs 0); '
                 f'n_negative/{len(cohort)}')
    ax.legend(loc='best')

    fig.suptitle('Per-metric speed correlation across NOF Day-1 cohort',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = OUT / 'fig_per_metric_speed.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')

    # Print summary
    print('\nPer-metric cohort summary (Real):')
    print(f'  {"Metric":<8} {"median r":>10} {"sign-/N":>10} {"Wilcoxon p":>14}')
    for m, mr, ng, p in zip(MEASURES, medians_r, sign_neg, pvals):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f'  {m:<8} {mr:>10.3f} {ng:>4}/{len(cohort):<5} '
              f'{p:>14.2e} {sig}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', default='1D',
                    choices=['1D', '2D', '3D', '4D', 'all'])
    args = ap.parse_args()
    print(f'=== Per-window analysis: NOF cohort (days={args.days}) ===')
    cohort = gather_cohort(args.days)
    cohort = [c for c in cohort if len(c['rqa_real']) > 0]
    print(f'\n{len(cohort)} sessions ready')
    plot_cohort_pca(cohort)
    per_metric_speed(cohort)


if __name__ == '__main__':
    main()
