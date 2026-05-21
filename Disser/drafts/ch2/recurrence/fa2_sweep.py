#!/usr/bin/env python
"""FA2 parameter sweep: maximise Mantel/KNN on a single representative session."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
from pathlib import Path
from itertools import product

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

SESSION = 'NOF_H32_1D'
DS = 5
PERCENTILE = 95.0
SUB = 2000
KNN = 10
NOF = DRIADA / 'DRIADA data/NOF/SynchronizedData26_v1'
RESULTS = Path(__file__).parent / 'results'


def metrics(layout, xy, seed=42):
    n = len(layout)
    rng = np.random.default_rng(seed)
    sub = rng.choice(n, min(SUB, n), replace=False)
    dL = pdist(layout[sub]); dP = pdist(xy[sub])
    r_pear, _ = stats.pearsonr(dL, dP)
    r_spear, _ = stats.spearmanr(dL, dP)
    _, _, disp = procrustes(xy, layout)
    k = KNeighborsRegressor(n_neighbors=KNN)
    perm = rng.permutation(n); h = n // 2
    k.fit(layout[perm[:h]], xy[perm[:h]])
    err = np.sqrt(((k.predict(layout[perm[h:]]) - xy[perm[h:]])**2)
                  .sum(axis=1)).mean()
    return dict(mant_p=r_pear, mant_s=r_spear, procr=disp, knn=err)


def main():
    from fa2_modified import ForceAtlas2

    # Load mean-matrix + behavior
    cache = RESULTS / SESSION / f'mean_matrix_ds{DS}_k50_exp_md3.npz'
    cached = np.load(cache, allow_pickle=True)
    mm = cached['mean_matrix']
    median_tau = int(np.median(cached['taus']))
    n = mm.shape[0]
    diag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < median_tau * 3
    m = mm.copy(); m[diag] = 0
    thr = np.percentile(m[m > 0], PERCENTILE)
    jrp = (m >= thr).astype(float)
    G = nx.from_scipy_sparse_array(sp.csr_matrix(jrp))
    print(f'{SESSION}: {G.number_of_edges()} edges, n={n}')

    exp = load_experiment_from_npz(NOF / f'{SESSION}_aligned.npz', verbose=False)
    n_full = exp.calcium.data.shape[1] // DS + (
        1 if exp.calcium.data.shape[1] % DS else 0)
    offset = n_full - n
    x = exp.dynamic_features['x'].data[::DS][offset:offset + n]
    y = exp.dynamic_features['y'].data[::DS][offset:offset + n]
    xy = np.column_stack([x, y])

    # Sweep grid
    grid = list(product(
        [0.1, 0.25, 0.5, 1.0, 2.0, 4.0],   # scalingRatio
        [0.5, 1.0, 2.0, 5.0],               # gravity
        [False, True],                       # outboundAttractionDistribution
        [False, True],                       # strongGravityMode
    ))
    print(f'{len(grid)} configurations\n')
    print(f'{"#":>2} {"sR":>5} {"grav":>5} {"oA":>5} {"sG":>5} '
          f'{"Mantel_s":>9} {"Mantel_p":>9} {"Procr":>7} {"KNN":>7}')
    print('-' * 70)
    rows = []
    for idx, (sR, gv, oA, sG) in enumerate(grid):
        t0 = time.time()
        fa2 = ForceAtlas2(outboundAttractionDistribution=oA,
                          barnesHutOptimize=True, barnesHutTheta=1.2,
                          scalingRatio=sR, gravity=gv,
                          strongGravityMode=sG, verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
        layout = np.array([pos[i] for i in range(G.number_of_nodes())])
        mt = metrics(layout, xy)
        rows.append((sR, gv, oA, sG, mt))
        print(f'{idx:>2} {sR:>5.2f} {gv:>5.1f} {str(oA):>5} {str(sG):>5} '
              f'{mt["mant_s"]:>9.3f} {mt["mant_p"]:>9.3f} '
              f'{mt["procr"]:>7.3f} {mt["knn"]:>7.2f}'
              f'   ({time.time()-t0:.0f}s)', flush=True)

    print('\nTop 5 by Mantel Spearman:')
    rows_sorted = sorted(rows, key=lambda r: -r[4]['mant_s'])
    for sR, gv, oA, sG, mt in rows_sorted[:5]:
        print(f'  sR={sR}, gv={gv}, oA={oA}, sG={sG} -> '
              f'Mantel_s={mt["mant_s"]:.3f}, KNN={mt["knn"]:.2f} cm')

    print('\nTop 5 by KNN error (lower is better):')
    rows_sorted = sorted(rows, key=lambda r: r[4]['knn'])
    for sR, gv, oA, sG, mt in rows_sorted[:5]:
        print(f'  sR={sR}, gv={gv}, oA={oA}, sG={sG} -> '
              f'KNN={mt["knn"]:.2f} cm, Mantel_s={mt["mant_s"]:.3f}')


if __name__ == '__main__':
    main()
