#!/usr/bin/env python
"""
Full analysis on Day-1 sessions of all NOF/LNOF mice.

Two stages:
  1. Build real + shuffled mean_matrix caches for each 1D session
  2. Run window-RQA clustering (real + shuffled) -> cohort summary

Usage:
    python run_full_1d.py --stage cache       # ~2-3h, builds 38 caches
    python run_full_1d.py --stage cluster     # quick, uses caches
    python run_full_1d.py --stage all         # both, sequential
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import argparse
import time
import subprocess
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d

DRIADA_ROOT = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA_ROOT / 'src'))
sys.path.insert(0, str(DRIADA_ROOT / 'tools'))

from load_synchronized_experiments import load_experiment_from_npz
from driada.information.info_base import MultiTimeSeries

NOF_DIR = DRIADA_ROOT / 'DRIADA data' / 'NOF' / 'SynchronizedData26_v1'
LNOF_DIR = DRIADA_ROOT / 'DRIADA data' / 'LNOF' / 'aligned'
RESULTS = Path(__file__).parent / 'results'

DS = 5
K = 50
TAU_METHOD = 'exponential_fit'
MAX_DIM = 3
SMOOTH_SIGMA = 2.0


def list_1d_sessions(cohort='both'):
    """All Day-1 sessions. cohort: 'nof', 'lnof', or 'both'."""
    nof = sorted([p.stem.replace('_aligned', '')
                  for p in NOF_DIR.glob('NOF_*_1D_aligned.npz')])
    lnof = sorted([p.stem.replace('_aligned', '')
                   for p in LNOF_DIR.glob('LNOF_*_1D_aligned.npz')])
    if cohort == 'nof':
        return nof
    if cohort == 'lnof':
        return lnof
    return nof + lnof


def npz_path(session):
    if session.startswith('LNOF'):
        return LNOF_DIR / f'{session}_aligned.npz'
    return NOF_DIR / f'{session}_aligned.npz'


def cache_paths(session):
    out = RESULTS / session
    sfx = f'ds{DS}_k{K}_{TAU_METHOD[:3]}_md{MAX_DIM}'
    return (out / f'mean_matrix_{sfx}.npz',
            out / f'mean_matrix_shuffled_{sfx}.npz')


def build_one(session):
    """Build real + shuffled caches for one session."""
    path = npz_path(session)
    if not path.exists():
        print(f'  {session}: SKIP (npz not found)')
        return False
    real_cache, shuf_cache = cache_paths(session)
    real_cache.parent.mkdir(parents=True, exist_ok=True)

    if real_cache.exists() and shuf_cache.exists():
        print(f'  {session}: already cached')
        return True

    t0 = time.time()
    exp = load_experiment_from_npz(path, verbose=False)
    calcium = exp.calcium.data[:, ::DS]
    if SMOOTH_SIGMA > 0:
        calcium = np.array([gaussian_filter1d(c, SMOOTH_SIGMA) for c in calcium])
    n_neurons = calcium.shape[0]

    if not real_cache.exists():
        mts = MultiTimeSeries(calcium, discrete=False)
        pop = mts.population_recurrence_graph(
            method='mean', k=K, n_jobs=-1,
            tau_method=TAU_METHOD, max_dim=MAX_DIM)
        mm = pop.adj.toarray() if sp.issparse(pop.adj) else pop.adj
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
        np.savez_compressed(real_cache, mean_matrix=mm,
                            taus=taus, dims=dims, rqa_dicts=rqa_dicts)
        t1 = time.time() - t0
        print(f'  {session}: real built ({mm.shape[0]}pts, {n_neurons}nrs, {t1:.0f}s)')

    if not shuf_cache.exists():
        t1 = time.time()
        mts_s = exp.get_multicell_shuffled_calcium(return_array=False)
        shuf = mts_s.data[:, ::DS]
        if SMOOTH_SIGMA > 0:
            shuf = np.array([gaussian_filter1d(c, SMOOTH_SIGMA) for c in shuf])
        mts_s = MultiTimeSeries(shuf, discrete=False)
        pop_s = mts_s.population_recurrence_graph(
            method='mean', k=K, n_jobs=-1,
            tau_method=TAU_METHOD, max_dim=MAX_DIM)
        mm_s = pop_s.adj.toarray() if sp.issparse(pop_s.adj) else pop_s.adj
        np.savez_compressed(shuf_cache, mean_matrix=mm_s)
        t2 = time.time() - t1
        print(f'  {session}: shuf built ({t2:.0f}s, total {time.time()-t0:.0f}s)')

    return True


def stage_cache(cohort='both'):
    sessions = list_1d_sessions(cohort)
    print(f'\n=== STAGE 1: build caches for {len(sessions)} Day-1 sessions ({cohort}) ===')
    print(f'Parameters: ds={DS}, k={K}, tau_method={TAU_METHOD}, max_dim={MAX_DIM}')
    t_total = time.time()
    ok, fail = 0, 0
    for i, s in enumerate(sessions, 1):
        print(f'\n[{i}/{len(sessions)}] {s}')
        try:
            if build_one(s):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f'  {s}: FAILED — {type(e).__name__}: {e}')
            fail += 1
    print(f'\n=== STAGE 1 done: {ok} cached, {fail} failed, '
          f'{(time.time()-t_total)/60:.1f} min ===')


def stage_cluster():
    """Run window-RQA clustering (real + shuffled) via cluster_windows_by_rqa.py."""
    print('\n=== STAGE 2: window-RQA clustering ===')
    here = Path(__file__).parent
    script = here / 'cluster_windows_by_rqa.py'

    # Real
    print('\n--- Real ---')
    subprocess.run([sys.executable, '-u', str(script),
                    '--batch', '--k', str(K)],
                   cwd=here, check=False)

    # Shuffled
    print('\n--- Shuffled ---')
    subprocess.run([sys.executable, '-u', str(script),
                    '--batch', '--shuffled', '--k', str(K)],
                   cwd=here, check=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', default='all',
                   choices=['cache', 'cluster', 'all'])
    p.add_argument('--cohort', default='both',
                   choices=['nof', 'lnof', 'both'])
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage in ('cache', 'all'):
        stage_cache(args.cohort)
    if args.stage in ('cluster', 'all'):
        stage_cluster()


if __name__ == '__main__':
    main()
