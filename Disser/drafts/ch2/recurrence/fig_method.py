#!/usr/bin/env python
"""
Figure 1 of section 2.x: method illustration.

Panels:
  A — calcium traces of several neurons in a time window (input)
  B — cutout of population mean-matrix (continuous) — what the method
      produces from N traces
  C — same cutout binarized at pct95 — the JRP used for analysis,
      with recurrent off-diagonal structures visible
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import label, binary_dilation

DRIADA = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA / 'src'))
sys.path.insert(0, str(DRIADA / 'tools'))
from load_synchronized_experiments import load_experiment_from_npz

SESSION = 'NOF_H32_1D'
DS = 5
FPS_EFF = 4.0
PERCENTILE = 95.0
CUT_START = 800     # time index in mean-matrix coordinates
CUT_SIZE = 200      # 200 / 4 fps = 50 s
N_TRACES = 5        # number of example traces to show

NOF = DRIADA / 'DRIADA data/NOF/SynchronizedData26_v1'
HERE = Path(__file__).parent
RESULTS = HERE / 'results'
OUT = RESULTS / 'window_rqa_clustering'


def main():
    cache = RESULTS / SESSION / f'mean_matrix_ds{DS}_k50_exp_md3.npz'
    cached = np.load(cache, allow_pickle=True)
    mm = cached['mean_matrix']
    taus = cached['taus']
    median_tau = int(np.median(taus))
    n = mm.shape[0]
    print(f'{SESSION}: n={n}, median_tau={median_tau}')

    # Mask diagonal and compute pct95 on full matrix
    diag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < median_tau * 3
    m = mm.copy(); m[diag] = 0
    thr = np.percentile(m[m > 0], PERCENTILE) if np.any(m > 0) else 0.0
    print(f'pct{PERCENTILE} threshold = {thr:.4f}')

    # Cutout of mean-matrix and its binarized version
    i0, i1 = CUT_START, CUT_START + CUT_SIZE
    cut = m[i0:i1, i0:i1]
    cut_bin = (cut >= thr).astype(float)
    print(f'cutout RR = {cut_bin.sum() / cut.size:.3f}')

    t_axis = np.arange(i0, i1) / FPS_EFF

    # Figure: 2 equal-size panels with breathing room between them
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.subplots_adjust(wspace=0.35)

    # Panel A (left): continuous mean-matrix cutout (was B)
    axA = axes[0]
    vmax = np.percentile(cut[cut > 0], 99) if np.any(cut > 0) else 1
    imA = axA.imshow(cut, cmap='viridis', aspect='equal', origin='lower',
                     extent=[t_axis[0], t_axis[-1], t_axis[0], t_axis[-1]],
                     vmin=0, vmax=vmax, interpolation='none')
    axA.set_xlabel('time (s)')
    axA.set_ylabel('time (s)')
    axA.set_title('A. Population mean-matrix $M_{t_1,t_2}$',
                  fontsize=11, loc='left')
    # use axes_locatable so the colorbar does not shrink panel A
    divA = make_axes_locatable(axA)
    cax_A = divA.append_axes('right', size='4%', pad=0.06)
    cb = fig.colorbar(imA, cax=cax_A)
    cb.set_label('fraction of recurring neurons', fontsize=9)

    # Panel B (right): binarized JRP, with recurrent regions outlined
    axB = axes[1]
    # base: light background
    axB.imshow(cut_bin, cmap='Greys', aspect='equal', origin='lower',
               extent=[t_axis[0], t_axis[-1], t_axis[0], t_axis[-1]],
               vmin=0, vmax=1, interpolation='none')

    # Detect & outline dense recurrent regions
    dil = binary_dilation(cut_bin.astype(bool), iterations=2)
    labels, n_lab = label(dil)
    min_area = 200   # pixels in dilated component
    bboxes = []
    for k in range(1, n_lab + 1):
        mask = labels == k
        if mask.sum() < min_area:
            continue
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        # Skip components touching the diagonal band (uninteresting)
        ci = (y0 + y1) / 2; cj = (x0 + x1) / 2
        if abs(ci - cj) < median_tau * 5:
            continue
        bboxes.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))

    for (x0, y0, w, h) in bboxes:
        rect = patches.Rectangle(
            (i0 / FPS_EFF + x0 / FPS_EFF, i0 / FPS_EFF + y0 / FPS_EFF),
            w / FPS_EFF, h / FPS_EFF,
            linewidth=1.3, edgecolor='#d62728', facecolor='none')
        axB.add_patch(rect)

    axB.set_xlabel('time (s)')
    axB.set_ylabel('time (s)')
    axB.set_title(f'B. Binarized JRP (top {int(100-PERCENTILE)}%) '
                  f'with outlined recurrent regions',
                  fontsize=11, loc='left')
    axB.set_xlim(t_axis[0], t_axis[-1])
    axB.set_ylim(t_axis[0], t_axis[-1])
    axB.set_aspect('equal')
    # reserve same right-side strip as colorbar slot of panel A to align widths
    divB = make_axes_locatable(axB)
    cax_B = divB.append_axes('right', size='4%', pad=0.06)
    cax_B.axis('off')
    print(f'Outlined {len(bboxes)} off-diagonal recurrent regions')

    fig.suptitle(
        f'Population mean-matrix and its binarization '
        f'(session {SESSION.split("_")[1]}, {CUT_SIZE} samples = '
        f'{CUT_SIZE/FPS_EFF:.0f} s)',
        fontsize=12, y=0.995)
    fig.savefig(OUT / 'fig_method.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUT / "fig_method.png"}')


if __name__ == '__main__':
    main()
