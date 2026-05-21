#!/usr/bin/env python
"""
Overlay RQA-windows on FA2 layout of the same population graph.

Shows the connection: high-DET windows sit in compact area of state-space
(animal stationary in arena), low-DET windows trace long paths through it
(animal moving across arena). One object — two readings.

Per-session figure:
  Panel A: layout, all nodes (colored by time)
  Panel B: layout + window traces colored by RQA cluster (high-DET vs low-DET)
  Panel C: arena trajectory, same color coding
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DRIADA = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA / 'src'))
sys.path.insert(0, str(DRIADA / 'tools'))
from load_synchronized_experiments import load_experiment_from_npz

DS = 5
FPS_EFF = 4.0
WIN_SEC = 25.0
OVERLAP = 0.5
PERCENTILE = 95.0

NOF = DRIADA / 'DRIADA data/NOF/SynchronizedData26_v1'
HERE = Path(__file__).parent
RESULTS = HERE / 'results'
OUT = RESULTS / 'window_rqa_clustering'
DATA = OUT / 'data'


def build_fa2(jrp_sparse, seed=42):
    from fa2_modified import ForceAtlas2
    G = nx.from_scipy_sparse_array(jrp_sparse)
    fa2 = ForceAtlas2(outboundAttractionDistribution=True,
                      barnesHutOptimize=True, barnesHutTheta=1.2,
                      scalingRatio=2.0, gravity=1.0, verbose=False)
    pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
    layout = np.array([pos[i] for i in range(G.number_of_nodes())])
    return layout, G


def process_session(session):
    cache = RESULTS / session / f'mean_matrix_ds{DS}_k50_exp_md3.npz'
    pw_path = DATA / f'per_window_{session}.npz'
    if not cache.exists() or not pw_path.exists():
        return None
    print(f'  Loading {session}...')
    r = np.load(cache, allow_pickle=True)
    mm = r['mean_matrix']
    taus = r['taus']
    median_tau = int(np.median(taus))
    n = mm.shape[0]

    # Same binarization as RQA pipeline (pct95)
    diag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < median_tau * 3
    m = mm.copy(); m[diag] = 0
    thr = np.percentile(m[m > 0], PERCENTILE) if np.any(m > 0) else 0.0
    jrp = (m >= thr).astype(float)

    # FA2 layout — same threshold as geometry pct95
    t0 = time.time()
    layout, G = build_fa2(sp.csr_matrix(jrp))
    print(f'  FA2 layout: {G.number_of_edges()} edges, {time.time()-t0:.1f}s')

    # Window RQA → re-cluster (KMeans, k=2)
    pw = np.load(pw_path, allow_pickle=True)
    rqa = pw['rqa_real']  # (n_win, 7)
    speed = pw['speed_real']
    n_win = rqa.shape[0]

    X = StandardScaler().fit_transform(rqa)
    km = KMeans(n_clusters=2, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    # cluster 0 = high-DET (column 0 is DET)
    det_means = [rqa[labels == c, 0].mean() for c in (0, 1)]
    if det_means[1] > det_means[0]:
        labels = 1 - labels
    print(f'  RQA clusters: high-DET={int((labels==0).sum())}, '
          f'low-DET={int((labels==1).sum())}')

    # Window centroids in layout (mean over window's frames)
    win_size = int(WIN_SEC * FPS_EFF)
    step = int(win_size * (1 - OVERLAP))
    centroids = np.zeros((n_win, 2))
    win_traces = []   # list of (T_win, 2) for line plotting
    for wi in range(n_win):
        i0 = wi * step
        i1 = min(i0 + win_size, n)
        pts = layout[i0:i1]
        centroids[wi] = pts.mean(0)
        win_traces.append(pts)

    # Arena (x, y), aligned to layout indices
    exp = load_experiment_from_npz(NOF / f'{session}_aligned.npz', verbose=False)
    n_full = exp.calcium.data.shape[1] // DS + (
        1 if exp.calcium.data.shape[1] % DS else 0)
    offset = n_full - n
    x = exp.dynamic_features['x'].data[::DS][offset:offset + n]
    y = exp.dynamic_features['y'].data[::DS][offset:offset + n]

    return dict(layout=layout, labels=labels, centroids=centroids,
                win_traces=win_traces, x=x, y=y, speed=speed,
                session=session, step=step, win_size=win_size)


def plot_session(d):
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 16))
    layout = d['layout']
    labels = d['labels']
    centroids = d['centroids']
    win_traces = d['win_traces']

    colors = {0: '#cc2222', 1: '#888888'}   # high-DET red, low-DET gray
    labels_text = {0: 'high-DET (regular)', 1: 'low-DET (exploratory)'}

    # Panel A: layout colored by time
    ax = axes[0]
    t_idx = np.arange(len(layout))
    sc = ax.scatter(layout[:, 0], layout[:, 1], c=t_idx, cmap='plasma',
                    s=3, alpha=0.6, edgecolor='none')
    ax.set_title('A. Layout colored by time', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='frame index')

    # Panel B: layout + per-window centroids + sparse traces
    ax = axes[1]
    ax.scatter(layout[:, 0], layout[:, 1], c='lightgray', s=2, alpha=0.3,
               edgecolor='none', zorder=1)
    # window traces
    for tr, l in zip(win_traces, labels):
        ax.plot(tr[:, 0], tr[:, 1], color=colors[l], lw=0.5, alpha=0.45,
                zorder=2)
    # centroids on top
    for c in (0, 1):
        m = labels == c
        ax.scatter(centroids[m, 0], centroids[m, 1], c=colors[c], s=35,
                   edgecolor='black', linewidth=0.4, alpha=0.85,
                   label=f'{labels_text[c]} (n={int(m.sum())})', zorder=3)
    ax.set_title('B. Layout + RQA-window traces and centroids', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')

    # Panel C: arena trajectory, same color coding
    ax = axes[2]
    win_centers_t = np.array([wi * d['step'] + d['win_size'] // 2
                              for wi in range(len(labels))])
    win_centers_t = np.clip(win_centers_t, 0, len(d['x']) - 1)
    # arena background — full trajectory in light gray
    ax.plot(d['x'], d['y'], color='lightgray', lw=0.4, alpha=0.7, zorder=1)
    # window centroids in arena = mean (x, y) over each window
    for wi, l in enumerate(labels):
        i0 = wi * d['step']
        i1 = min(i0 + d['win_size'], len(d['x']))
        xc, yc = d['x'][i0:i1].mean(), d['y'][i0:i1].mean()
        ax.scatter(xc, yc, c=colors[l], s=35, edgecolor='black',
                   linewidth=0.4, alpha=0.85, zorder=3)
    ax.set_title('C. Arena: window centroids by RQA cluster', fontsize=11)
    ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
    ax.set_aspect('equal')

    fig.suptitle(f'{d["session"]}: one graph, two readings — geometry (layout) '
                 f'× dynamics (RQA window clusters)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT / f'layout_rqa_overlay_{d["session"]}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


def main():
    # Two representative sessions: strong geometry, strong RQA effect
    targets = ['NOF_H39_1D', 'NOF_H32_1D']
    for s in targets:
        d = process_session(s)
        if d is None:
            print(f'  SKIP {s} (no cache)')
            continue
        plot_session(d)


if __name__ == '__main__':
    main()
