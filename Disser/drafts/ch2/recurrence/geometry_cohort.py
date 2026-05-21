#!/usr/bin/env python
"""
Cohort-level geometry analysis on NOF Day-1 sessions.

For each session: build FA2 layouts of real and matched-density shuffled JRP,
compute Mantel/Procrustes/KNN-decoder metrics. Cohort summary: Wilcoxon
real vs shuffled per metric; sign-test.
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

DS = 5
TARGET_DEGREE = 12
PERCENTILE = 95.0          # alternative threshold mode
THRESHOLD_MODE = 'percentile'   # 'target_degree' or 'percentile'
N_SUBSAMPLE = 2000
KNN = 10
NOF = DRIADA / 'DRIADA data/NOF/SynchronizedData26_v1'
HERE = Path(__file__).parent
RESULTS = HERE / 'results'
OUT = RESULTS / 'window_rqa_clustering'
OUT.mkdir(parents=True, exist_ok=True)


def list_nof_1d():
    return sorted([p.stem.replace('_aligned', '')
                   for p in NOF.glob('NOF_*_1D_aligned.npz')])


def threshold_for_density(m_off, target_edges):
    """Threshold giving exactly target_edges edges."""
    flat = m_off[m_off > 0].ravel()
    if len(flat) <= 2 * target_edges:
        return 0.0
    return np.sort(flat)[-2 * target_edges]


def build_layout(jrp_sparse, seed=42):
    """FA2 layout with sweep-optimised parameters
    (sR=1.0, gravity=0.5; gives Mantel_s ≈ 0.5 on NOF_H32 vs 0.38 default)."""
    from fa2_modified import ForceAtlas2
    G = nx.from_scipy_sparse_array(jrp_sparse)
    fa2 = ForceAtlas2(outboundAttractionDistribution=True,
                      barnesHutOptimize=True, barnesHutTheta=1.2,
                      scalingRatio=1.0, gravity=0.5,
                      strongGravityMode=False, verbose=False)
    pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
    layout = np.array([pos[i] for i in range(G.number_of_nodes())])
    return layout, G


def metrics(layout, xy, rng_seed):
    n = len(layout)
    rng = np.random.default_rng(rng_seed)
    sub = rng.choice(n, min(N_SUBSAMPLE, n), replace=False)
    dL = pdist(layout[sub]); dP = pdist(xy[sub])
    r_pear, _ = stats.pearsonr(dL, dP)
    r_spear, _ = stats.spearmanr(dL, dP)
    _, _, disp = procrustes(xy, layout)
    knn = KNeighborsRegressor(n_neighbors=KNN)
    perm = rng.permutation(n)
    h = n // 2
    knn.fit(layout[perm[:h]], xy[perm[:h]])
    pred = knn.predict(layout[perm[h:]])
    err = np.sqrt(((pred - xy[perm[h:]])**2).sum(axis=1)).mean()
    return dict(mantel_p=r_pear, mantel_s=r_spear, procr=disp, knn_err=err)


def process_session(session):
    cache = RESULTS / session / f'mean_matrix_ds{DS}_k50_exp_md3.npz'
    shuf = RESULTS / session / f'mean_matrix_shuffled_ds{DS}_k50_exp_md3.npz'
    if not cache.exists() or not shuf.exists():
        return None
    r_mm = np.load(cache, allow_pickle=True)
    s_mm = np.load(shuf, allow_pickle=True)
    real_mm = r_mm['mean_matrix']
    shuf_mm = s_mm['mean_matrix']
    taus = r_mm['taus']
    median_tau = int(np.median(taus))
    n = min(real_mm.shape[0], shuf_mm.shape[0])
    real_mm = real_mm[:n, :n]; shuf_mm = shuf_mm[:n, :n]
    diag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < median_tau * 3
    m_r = real_mm.copy(); m_r[diag] = 0
    m_s = shuf_mm.copy(); m_s[diag] = 0

    # Real: threshold either by target degree or by percentile
    if THRESHOLD_MODE == 'target_degree':
        target_edges = int(TARGET_DEGREE * n / 2)
        thr_r = threshold_for_density(m_r, target_edges)
    else:  # percentile (same as RQA-clustering pipeline)
        thr_r = np.percentile(m_r[m_r > 0], PERCENTILE) if np.any(m_r > 0) else 0.0
    jrp_r = (m_r >= thr_r).astype(float)
    n_edges_r = int(jrp_r.sum() / 2)

    # Shuffled: matched density to real (so we compare structure at equal RR)
    thr_s = threshold_for_density(m_s, n_edges_r)
    jrp_s = (m_s >= thr_s).astype(float)
    n_edges_s = int(jrp_s.sum() / 2)

    # (x, y), aligned
    exp = load_experiment_from_npz(NOF / f'{session}_aligned.npz', verbose=False)
    n_full = exp.calcium.data.shape[1] // DS + (
        1 if exp.calcium.data.shape[1] % DS else 0)
    offset = n_full - n
    x = exp.dynamic_features['x'].data[::DS][offset:offset + n]
    y = exp.dynamic_features['y'].data[::DS][offset:offset + n]
    xy = np.column_stack([x, y])

    # Layouts
    t0 = time.time()
    layout_r, G_r = build_layout(sp.csr_matrix(jrp_r))
    layout_s, G_s = build_layout(sp.csr_matrix(jrp_s))
    layout_time = time.time() - t0

    # Metrics
    m_r_dict = metrics(layout_r, xy, rng_seed=42)
    m_s_dict = metrics(layout_s, xy, rng_seed=43)

    # Baselines
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    h = n // 2
    pred_mean = xy[perm[:h]].mean(0)
    err_baseline = np.sqrt(
        ((pred_mean - xy[perm[h:]])**2).sum(axis=1)).mean()

    return {
        'session': session, 'n': n, 'n_edges_r': n_edges_r,
        'n_edges_s': n_edges_s, 'layout_time': layout_time,
        'r_mantel_p': m_r_dict['mantel_p'],
        'r_mantel_s': m_r_dict['mantel_s'],
        'r_procr': m_r_dict['procr'],
        'r_knn_err': m_r_dict['knn_err'],
        's_mantel_p': m_s_dict['mantel_p'],
        's_mantel_s': m_s_dict['mantel_s'],
        's_procr': m_s_dict['procr'],
        's_knn_err': m_s_dict['knn_err'],
        'baseline_err': err_baseline,
    }


def main():
    sessions = list_nof_1d()
    print(f'=== Cohort geometry: {len(sessions)} NOF Day-1 sessions ===\n')
    rows = []
    t0 = time.time()
    for i, s in enumerate(sessions, 1):
        print(f'[{i}/{len(sessions)}] {s}', flush=True)
        try:
            r = process_session(s)
            if r is None:
                print(f'  SKIP (no cache)')
                continue
        except Exception as e:
            print(f'  ERROR: {type(e).__name__}: {e}')
            continue
        rows.append(r)
        print(f'  edges r/s={r["n_edges_r"]}/{r["n_edges_s"]}, '
              f'layout {r["layout_time"]:.0f}s, '
              f'Mantel_s r/s={r["r_mantel_s"]:+.2f}/{r["s_mantel_s"]:+.2f}, '
              f'KNN err r/s/baseline='
              f'{r["r_knn_err"]:.1f}/{r["s_knn_err"]:.1f}/{r["baseline_err"]:.1f}cm',
              flush=True)

    print(f'\nTotal cohort time: {(time.time()-t0)/60:.1f} min')

    # CSV (suffix encodes threshold mode for reproducibility)
    suffix = ('_k12' if THRESHOLD_MODE == 'target_degree'
              else f'_pct{int(PERCENTILE)}')
    csv = OUT / f'geometry_cohort{suffix}.csv'
    keys = list(rows[0].keys())
    with open(csv, 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(str(r[k]) for k in keys) + '\n')
    print(f'Saved CSV: {csv}')

    # Cohort tests
    arr = lambda key: np.array([r[key] for r in rows])
    print(f'\n=== Cohort summary (n={len(rows)} mice) ===\n')
    print(f'{"Metric":<22} {"Real median":>12} {"Shuf median":>12} '
          f'{"Sign R>S":>10} {"Wilcoxon p":>12}')
    for label, k_r, k_s, alt in [
        ('Mantel Pearson',   'r_mantel_p', 's_mantel_p', 'greater'),
        ('Mantel Spearman',  'r_mantel_s', 's_mantel_s', 'greater'),
        ('Procrustes disp.', 'r_procr',    's_procr',    'less'),
        ('KNN error (cm)',   'r_knn_err',  's_knn_err',  'less'),
    ]:
        a_r, a_s = arr(k_r), arr(k_s)
        med_r, med_s = np.median(a_r), np.median(a_s)
        diff = a_r - a_s
        if alt == 'greater':
            n_correct = int((diff > 0).sum())
        else:
            n_correct = int((diff < 0).sum())
        w = stats.wilcoxon(diff, alternative=alt).pvalue
        print(f'{label:<22} {med_r:>12.4f} {med_s:>12.4f} '
              f'{n_correct:>6}/{len(rows):<3} {w:>12.2e}')

    knn_r = arr('r_knn_err'); base = arr('baseline_err')
    improvement = (base - knn_r) / base * 100
    print(f'\nKNN improvement over random baseline: '
          f'median {np.median(improvement):.1f}%, '
          f'range [{improvement.min():.1f}, {improvement.max():.1f}]%')

    # Figure: forest plot
    n = len(rows)
    order = np.argsort(arr('r_knn_err'))
    labels = [rows[i]['session'].split('_')[1] for i in order]
    knn_r_s = arr('r_knn_err')[order]
    knn_s_s = arr('s_knn_err')[order]
    knn_b_s = arr('baseline_err')[order]
    m_r_s = arr('r_mantel_s')[order]
    m_s_s = arr('s_mantel_s')[order]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.35 * n + 2)))
    y = np.arange(n)
    # Panel A: KNN error
    ax = axes[0]
    ax.barh(y - 0.2, knn_r_s, height=0.38, color='#2CA02C',
            edgecolor='k', linewidth=0.4, label='Real')
    ax.barh(y + 0.2, knn_s_s, height=0.38, color='#FF7F0E',
            edgecolor='k', linewidth=0.4, label='Shuffled (matched)')
    ax.scatter(knn_b_s, y, marker='|', s=60, c='black',
               label='Random baseline')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('KNN decoder error (cm)')
    ax.set_title('A. KNN decoder error (real vs shuffled, matched density)',
                 fontsize=11, loc='left')
    ax.legend(loc='lower right', fontsize=9)
    ax.invert_yaxis()

    # Panel B: Mantel Spearman
    ax = axes[1]
    ax.barh(y - 0.2, m_r_s, height=0.38, color='#2CA02C',
            edgecolor='k', linewidth=0.4, label='Real')
    ax.barh(y + 0.2, m_s_s, height=0.38, color='#FF7F0E',
            edgecolor='k', linewidth=0.4, label='Shuffled')
    ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mantel Spearman r')
    ax.set_title('B. Mantel-like correlation', fontsize=11, loc='left')
    ax.legend(loc='lower right', fontsize=9)
    ax.invert_yaxis()

    fig.suptitle(f'NOF Day-1 cohort: spatial reconstruction from JRP layout '
                 f'({n} mice)', fontsize=13, y=1.0)
    fig.tight_layout()
    out = OUT / f'fig_cohort_geometry{suffix}.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {out}')


if __name__ == '__main__':
    main()
