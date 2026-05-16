#!/usr/bin/env python
"""
Parameter sweep for HD population recurrence: tau × m × theiler_window.

Measures circular correlation between spectral embedding angle and true HD.
Outputs summary table + heatmap.

Usage:
    python run_hd_sweep.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
import numpy as np
import scipy.sparse as sp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.manifold import SpectralEmbedding
from scipy.stats import pearsonr

DRIADA_ROOT = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA_ROOT / 'src'))

from driada.experiment.synthetic import generate_tuned_selectivity_exp
from driada.information.info_base import MultiTimeSeries

OUT = Path(__file__).parent / 'results' / 'synthetic_hd'
OUT.mkdir(parents=True, exist_ok=True)


def best_circ_r(coords, hd, n_dims=4):
    """Best circular correlation across all pairs of embedding dimensions."""
    best = 0
    best_pair = (0, 1)
    for d1 in range(n_dims):
        for d2 in range(d1 + 1, n_dims):
            ea = np.arctan2(coords[:, d2], coords[:, d1])
            rc, _ = pearsonr(np.cos(ea), np.cos(hd))
            rs, _ = pearsonr(np.sin(ea), np.sin(hd))
            cr = np.sqrt(rc**2 + rs**2)
            if cr > best:
                best = cr
                best_pair = (d1, d2)
    return best, best_pair


def run_one(calcium, hd_full, n_frames, tau, m, theiler, k=50):
    """Run population recurrence + spectral embedding for one parameter set."""
    mts = MultiTimeSeries(calcium, discrete=False)

    kwargs = dict(method='mean', k=k, n_jobs=-1, theiler_window=theiler)
    if tau != 'auto':
        kwargs['tau'] = tau
    if m != 'auto':
        kwargs['m'] = m

    pop_rg = mts.population_recurrence_graph(**kwargs)
    mm = pop_rg.adj.toarray() if sp.issparse(pop_rg.adj) else pop_rg.adj
    min_n = mm.shape[0]
    offset = n_frames - min_n
    hd = hd_full[offset:offset + min_n]

    # Actual embedding params
    if tau == 'auto':
        actual_taus = np.array([ts._recurrence_tau[1] for ts in mts.ts_list])
        actual_dims = np.array([ts._recurrence_embedding_dim[1]
                                for ts in mts.ts_list])
        emb_win = int(np.median(actual_taus * (actual_dims - 1)))
    else:
        m_val = m if m != 'auto' else 1
        emb_win = tau * (m_val - 1)

    # Minimal diagonal mask
    diag_mask = (np.abs(np.arange(min_n)[:, None] - np.arange(min_n)[None, :])
                 < 3)
    md = mm.copy()
    md[diag_mask] = 0

    # Spectral embedding
    se = SpectralEmbedding(n_components=4, affinity='precomputed',
                           random_state=42)
    coords = se.fit_transform(md)
    cr, pair = best_circ_r(coords, hd)

    return cr, pair, emb_win, min_n


def main():
    t_total = time.time()

    # Generate data once
    print('Generating 200 HD neurons, 600s @ 5fps...')
    exp = generate_tuned_selectivity_exp(
        population=[{'name': 'hd', 'count': 200,
                     'features': ['head_direction']}],
        duration=600, fps=5, seed=42, verbose=False)
    calcium = exp.calcium.data
    hd_full = exp.dynamic_features['head_direction'].data
    n_frames = calcium.shape[1]

    # Parameter grid
    grid = [
        # (tau, m, theiler)
        (1,     1, 0),
        (1,     1, 5),
        (1,     1, 10),
        (1,     1, 20),
        (2,     2, 0),
        (2,     2, 5),
        (2,     2, 10),
        (2,     2, 20),
        (3,     2, 0),
        (3,     2, 5),
        (3,     2, 10),
        (3,     2, 20),
        (5,     2, 0),
        (5,     2, 5),
        (5,     2, 10),
        (5,     2, 20),
        (5,     3, 0),
        (5,     3, 5),
        (5,     3, 10),
        (5,     3, 20),
        (10,    3, 0),
        (10,    3, 5),
        (10,    3, 20),
        (10,    5, 0),
        (10,    5, 5),
        (10,    5, 20),
        ('auto', 'auto', 0),
        ('auto', 'auto', 5),
        ('auto', 'auto', 20),
        ('auto', 'auto', 'auto'),
    ]

    results = []
    print(f'\n{"tau":>5} {"m":>5} {"theiler":>7} {"emb_win":>8} '
          f'{"circ_r":>7} {"best_pair":>10} {"time":>6}')
    print('-' * 55)

    import warnings
    warnings.filterwarnings('ignore')

    for tau, m, tw in grid:
        t0 = time.time()
        cr, pair, ew, min_n = run_one(calcium, hd_full, n_frames, tau, m, tw)
        elapsed = time.time() - t0
        results.append({
            'tau': tau, 'm': m, 'theiler': tw,
            'emb_win': ew, 'circ_r': cr, 'pair': pair,
            'min_n': min_n, 'time': elapsed,
        })
        print(f'{str(tau):>5} {str(m):>5} {str(tw):>7} {ew:>8} '
              f'{cr:>7.3f} {str(pair):>10} {elapsed:>5.1f}s')

    # ---- Heatmap ----
    # Rows: (tau, m) combos; Cols: theiler values
    row_keys = []
    seen = set()
    for r in results:
        key = (str(r['tau']), str(r['m']))
        if key not in seen:
            row_keys.append(key)
            seen.add(key)

    col_keys = sorted(set(str(r['theiler']) for r in results),
                      key=lambda x: 9999 if x == 'auto' else int(x))

    heat = np.full((len(row_keys), len(col_keys)), np.nan)
    ew_labels = {}
    for r in results:
        ri = row_keys.index((str(r['tau']), str(r['m'])))
        ci = col_keys.index(str(r['theiler']))
        heat[ri, ci] = r['circ_r']
        ew_labels[(str(r['tau']), str(r['m']))] = r['emb_win']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heat, cmap='RdYlGn', vmin=0, vmax=0.85, aspect='auto')
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if not np.isnan(heat[i, j]):
                color = 'white' if heat[i, j] < 0.35 else 'black'
                ax.text(j, i, f'{heat[i, j]:.3f}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)

    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels([f'th={t}' for t in col_keys])
    ax.set_yticks(range(len(row_keys)))
    y_labels = []
    for k in row_keys:
        ew = ew_labels.get(k, '?')
        y_labels.append(f'τ={k[0]}, m={k[1]}  (win={ew})')
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Theiler window (samples)', fontsize=12)
    ax.set_ylabel('Embedding parameters', fontsize=12)
    ax.set_title('Circular correlation: SE angle vs true HD\n'
                 '200 HD neurons, 600s @ 5fps, k=50', fontsize=13)
    plt.colorbar(im, ax=ax, label='circ_r', shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_parameter_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {OUT / "fig_parameter_sweep.png"}')
    print(f'Total: {time.time() - t_total:.0f}s')


if __name__ == '__main__':
    main()
