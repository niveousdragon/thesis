#!/usr/bin/env python
"""
Cross-validate ED (effective dimension, sec 2.4) vs RQA-PC1 (sec 2.x)
on the same 25-s windows of NOF Day-1 sessions.

Pipeline per session:
  1. Load calcium, downsample to match RQA pipeline (ds=5 -> 4 Hz)
  2. Slide 25-s windows (same boundaries as per_window_*.npz)
  3. For each window: ED via driada.dimensionality.eff_dim (with MP correction)
  4. Pair with RQA-PC1 (from cohort PCA) and DET (RQA metric 0)

Outputs:
  data/ed_per_window_<session>.npz
  fig_ed_vs_rqa.png    (3 panels: per-session scatter, cohort-r, ED vs speed)
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
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DRIADA_ROOT = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA_ROOT / 'src'))
sys.path.insert(0, str(DRIADA_ROOT / 'tools'))
from load_synchronized_experiments import load_experiment_from_npz
from driada.dimensionality.effective import eff_dim

NOF_DIR = DRIADA_ROOT / 'DRIADA data' / 'NOF' / 'SynchronizedData26_v1'
HERE = Path(__file__).parent
RESULTS = HERE / 'results'
OUT = RESULTS / 'window_rqa_clustering'
DATA = OUT / 'data'
DATA.mkdir(parents=True, exist_ok=True)

DS = 5
WIN_SEC = 25.0
OVERLAP = 0.5
SMOOTH = 2.0
MEASURES = ['DET', 'LAM', 'ENTR', 'TT', 'L_mean', 'L_max', 'DIV']
FPS_EFF = 20.0 / DS   # 4 Hz


def compute_ed_windows(session):
    """ED on the same 25-s windows as RQA. Aligned with per_window_<session>.npz."""
    per_win_path = DATA / f'per_window_{session}.npz'
    if not per_win_path.exists():
        return None
    npz = NOF_DIR / f'{session}_aligned.npz'
    if not npz.exists():
        return None
    exp = load_experiment_from_npz(npz, verbose=False)

    # ED is computed on RAW dF/F (no gaussian smoothing).
    # RQA used smoothing for embedding stability, but ED relies on the
    # full covariance spectrum, which gaussian smoothing destroys.
    calcium = exp.calcium.data[:, ::DS]
    n_neurons, n_frames = calcium.shape

    # Load RQA cache to recover embedding-loss offset.
    # RQA pipeline: min_n = mean_matrix.shape[0] = n_frames - embed_loss.
    # Window wi covers indices [wi*step, wi*step + win_size] of the trimmed
    # (last min_n) series. Trimming happens at the BEGINNING (embedding loss
    # removes (m-1)*tau samples from the start).
    cache_path = (RESULTS / session
                  / f'mean_matrix_ds{DS}_k50_exp_md3.npz')
    cached = np.load(cache_path, allow_pickle=True)
    min_n = cached['mean_matrix'].shape[0]
    offset = n_frames - min_n   # samples trimmed from the start

    pw = np.load(per_win_path, allow_pickle=True)
    n_rqa_windows = pw['rqa_real'].shape[0]
    win_size = int(WIN_SEC * FPS_EFF)
    step = int(win_size * (1 - OVERLAP))

    ed_vals = np.full(n_rqa_windows, np.nan)
    for wi in range(n_rqa_windows):
        i0 = offset + wi * step
        i1 = i0 + win_size
        if i1 > n_frames:
            break
        # eff_dim expects (samples, features) — i.e. (time, neurons)
        window_data = calcium[:, i0:i1].T
        # MP correction expects T >= N for stable estimate; use default mode
        try:
            ed_vals[wi] = eff_dim(window_data, enable_correction=True)
        except Exception:
            try:
                ed_vals[wi] = eff_dim(window_data, enable_correction=False)
            except Exception:
                ed_vals[wi] = np.nan

    return {
        'session': session,
        'ed': ed_vals,
        'rqa': pw['rqa_real'],
        'speed': pw['speed_real'],
        'rqa_shuf': pw['rqa_shuf'],
        'speed_shuf': pw['speed_shuf'],
    }


def list_sessions_with_pw():
    return sorted([p.stem.replace('per_window_', '')
                   for p in DATA.glob('per_window_NOF_*_1D.npz')])


def main():
    sessions = list_sessions_with_pw()
    print(f'Computing ED on {len(sessions)} sessions...')
    cohort = []
    for s in sessions:
        cache = DATA / f'ed_{s}.npz'
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            cohort.append({'session': s, 'ed': d['ed'],
                           'rqa': d['rqa'], 'speed': d['speed'],
                           'rqa_shuf': d['rqa_shuf'],
                           'speed_shuf': d['speed_shuf']})
            print(f'  {s}: loaded')
            continue
        t0 = time.time()
        r = compute_ed_windows(s)
        if r is None:
            print(f'  {s}: SKIP')
            continue
        np.savez_compressed(cache, ed=r['ed'], rqa=r['rqa'],
                            speed=r['speed'], rqa_shuf=r['rqa_shuf'],
                            speed_shuf=r['speed_shuf'])
        cohort.append(r)
        n_valid = np.sum(~np.isnan(r['ed']))
        print(f'  {s}: {n_valid}/{len(r["ed"])} valid windows ({time.time()-t0:.1f}s)')

    # Build cohort-wide PCA fit (same protocol as per_window_analysis.py)
    Xall = np.vstack([c['rqa'] for c in cohort])
    scaler = StandardScaler().fit(Xall)
    pca = PCA(n_components=2).fit(scaler.transform(Xall))
    print(f'\nPC1 expl var: {pca.explained_variance_ratio_[0]:.3f}')

    # Per-session correlations
    rows = []
    for c in cohort:
        valid = ~np.isnan(c['ed'])
        ed = c['ed'][valid]
        sp = c['speed'][valid]
        pc1 = pca.transform(scaler.transform(c['rqa']))[valid, 0]
        det = c['rqa'][valid, MEASURES.index('DET')]
        if len(ed) < 5:
            continue
        r_pc1, p_pc1 = stats.pearsonr(ed, pc1)
        r_det, p_det = stats.pearsonr(ed, det)
        r_sp,  p_sp  = stats.pearsonr(ed, sp)
        rows.append({'session': c['session'], 'n': len(ed),
                     'r_ed_pc1': r_pc1, 'p_ed_pc1': p_pc1,
                     'r_ed_det': r_det, 'p_ed_det': p_det,
                     'r_ed_speed': r_sp, 'p_ed_speed': p_sp,
                     'ed': ed, 'pc1': pc1, 'det': det, 'speed': sp})

    # Cohort tests on per-session r values
    arr_pc1 = np.array([r['r_ed_pc1'] for r in rows])
    arr_det = np.array([r['r_ed_det'] for r in rows])
    arr_sp  = np.array([r['r_ed_speed'] for r in rows])

    print(f'\n--- Per-session correlations (n={len(rows)} sessions) ---')
    for name, arr in [('ED vs RQA-PC1', arr_pc1),
                      ('ED vs DET',     arr_det),
                      ('ED vs speed',   arr_sp)]:
        med = np.median(arr)
        # one-sided: ED is high on stops, PC1/DET are high on stops too -> positive
        # ED vs speed: negative
        alt = 'greater' if 'speed' not in name else 'less'
        w = stats.wilcoxon(arr, alternative=alt).pvalue
        # sign-correct count
        n_correct = (arr > 0).sum() if alt == 'greater' else (arr < 0).sum()
        print(f'  {name:<16} median r = {med:+.3f},  '
              f'{n_correct}/{len(arr)} expected-sign,  '
              f'Wilcoxon p = {w:.2e}')

    # ===========================================================
    # Figure
    # ===========================================================
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # Panel A: pooled scatter ED vs PC1 colored by mouse
    ax = axes[0]
    cmap = plt.cm.tab20
    for i, c in enumerate(rows):
        ax.scatter(c['pc1'], c['ed'], s=10, color=cmap(i / max(1, len(rows))),
                   alpha=0.55, edgecolor='none')
    pooled_pc1 = np.concatenate([r['pc1'] for r in rows])
    pooled_ed  = np.concatenate([r['ed']  for r in rows])
    r_pooled, p_pooled = stats.pearsonr(pooled_pc1, pooled_ed)
    ax.set_xlabel('RQA-PC1 (regularity index)')
    ax.set_ylabel('ED (effective dimension)')
    ax.set_title(f'A. Per-window ED vs RQA-PC1\n'
                 f'pooled r={r_pooled:+.2f}, p={p_pooled:.1e}',
                 fontsize=11, loc='left')

    # Panel B: per-session r distributions
    ax = axes[1]
    bp = ax.boxplot([arr_pc1, arr_det, arr_sp], labels=['vs PC1', 'vs DET', 'vs speed'],
                    patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor='#e0e0e0', linewidth=0.6),
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', ms=3))
    # overlay points
    for i, arr in enumerate([arr_pc1, arr_det, arr_sp]):
        ax.scatter(np.full(len(arr), i + 1) +
                   (np.random.RandomState(i).rand(len(arr)) - 0.5) * 0.18,
                   arr, color='#cc3344', s=22, alpha=0.85,
                   edgecolor='black', linewidth=0.4, zorder=3)
    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.7)
    ax.set_ylabel('Per-mouse Pearson r')
    ax.set_title(f'B. Per-session r ({len(rows)} mice)',
                 fontsize=11, loc='left')

    # Panel C: ED on a single representative session, color = speed
    # pick the session with strongest ED-PC1 correlation
    best = max(rows, key=lambda r: r['r_ed_pc1'])
    ax = axes[2]
    sc = ax.scatter(best['pc1'], best['ed'], c=best['speed'],
                    cmap='plasma', s=22, alpha=0.85, edgecolor='black',
                    linewidth=0.3)
    ax.set_xlabel('RQA-PC1 (regularity index)')
    ax.set_ylabel('ED')
    ax.set_title(
        f'C. Example: {best["session"].split("_")[1]}\n'
        f'r={best["r_ed_pc1"]:+.2f}, n={best["n"]} windows',
        fontsize=11, loc='left')
    plt.colorbar(sc, ax=ax, label='window speed', shrink=0.85)

    fig.suptitle('Cross-validation: effective dimension (sec 2.4) vs RQA regularity (sec 2.x)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_fig = OUT / 'fig_ed_vs_rqa.png'
    fig.savefig(out_fig, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_fig}')

    # CSV
    csv = OUT / 'ed_vs_rqa.csv'
    with open(csv, 'w', encoding='utf-8') as f:
        f.write('session,n,r_ed_pc1,p_ed_pc1,r_ed_det,p_ed_det,r_ed_speed,p_ed_speed\n')
        for r in rows:
            f.write(f'{r["session"]},{r["n"]},'
                    f'{r["r_ed_pc1"]:.4f},{r["p_ed_pc1"]:.4e},'
                    f'{r["r_ed_det"]:.4f},{r["p_ed_det"]:.4e},'
                    f'{r["r_ed_speed"]:.4f},{r["p_ed_speed"]:.4e}\n')
    print(f'Saved: {csv}')


if __name__ == '__main__':
    main()
