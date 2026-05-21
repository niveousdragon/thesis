#!/usr/bin/env python
"""
Inspect NOF_H26_1D — the only anti-direction session.

Hypothesis: high-DET windows in the first ~100s reflect stereotyped fast
locomotion (e.g., thigmotaxis along the wall), not stops.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DRIADA = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA / 'src'))
sys.path.insert(0, str(DRIADA / 'tools'))
from load_synchronized_experiments import load_experiment_from_npz

SESSION = 'NOF_H26_1D'
DS = 5  # match clustering pipeline


def main():
    exp = load_experiment_from_npz(
        DRIADA / f'DRIADA data/NOF/SynchronizedData26_v1/{SESSION}_aligned.npz',
        verbose=False)

    # behavioral data, downsampled to match analysis
    x = exp.dynamic_features['x'].data[::DS]
    y = exp.dynamic_features['y'].data[::DS]
    speed = exp.dynamic_features['speed'].data[::DS]
    walls = exp.dynamic_features['walls'].data[::DS]
    walk = exp.dynamic_features['walk'].data[::DS]
    bd = exp.dynamic_features['bodydirection'].data[::DS]
    n = len(x)
    fps_eff = 20 / DS  # 4 Hz
    t = np.arange(n) / fps_eff

    # session split: first 100 s = "stereotype window" vs rest
    cut = int(100 * fps_eff)

    print(f'{SESSION}: {n} frames @ {fps_eff} Hz ({n/fps_eff:.0f} s)')
    print(f'  Mean speed:   {speed.mean():.2f} (early {speed[:cut].mean():.2f} '
          f'vs late {speed[cut:].mean():.2f})')
    print(f'  Walls frac:   early {walls[:cut].mean():.2f} '
          f'vs late {walls[cut:].mean():.2f}')
    print(f'  Walking frac: early {walk[:cut].mean():.2f} '
          f'vs late {walk[cut:].mean():.2f}')
    # body-direction variability: lower std = more stereotyped
    print(f'  BD circular std: early {np.angle(np.exp(1j*bd[:cut]).mean(), deg=False):.2f}  '
          f'(R={np.abs(np.exp(1j*bd[:cut]).mean()):.2f}), '
          f'late R={np.abs(np.exp(1j*bd[cut:]).mean()):.2f}')

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # A: trajectory colored by time (early = warm, late = cold)
    ax = axes[0, 0]
    sc = ax.scatter(x, y, c=t, cmap='viridis', s=4, alpha=0.5)
    ax.scatter(x[:cut], y[:cut], facecolor='none', edgecolor='red', s=15,
               linewidth=0.5, label=f'first {int(cut/fps_eff)}s')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    ax.set_title('Trajectory (color=time)'); ax.legend()
    plt.colorbar(sc, ax=ax, label='t (s)', shrink=0.7)

    # B: trajectory colored by speed
    ax = axes[0, 1]
    sc = ax.scatter(x, y, c=speed, cmap='plasma', s=4, alpha=0.6,
                    vmin=0, vmax=np.percentile(speed, 99))
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    ax.set_title('Trajectory (color=speed)')
    plt.colorbar(sc, ax=ax, label='speed', shrink=0.7)

    # C: speed timeline + walls/walk + cluster windows
    ax = axes[1, 0]
    ax.plot(t, speed, 'k-', lw=0.8, alpha=0.8, label='speed')
    ax.axvspan(0, 100, alpha=0.15, color='red', label='early 0–100s')
    ax.set_xlabel('time (s)'); ax.set_ylabel('speed')
    ax2 = ax.twinx()
    ax2.fill_between(t, 0, walls, alpha=0.3, color='blue', label='walls')
    ax2.set_ylabel('walls (0/1)', color='blue')
    ax.set_title('Speed & wall occupancy timeline'); ax.legend(loc='upper right')

    # D: body direction over time — stereotype check
    ax = axes[1, 1]
    ax.plot(t, np.unwrap(bd), 'k-', lw=0.8, alpha=0.8)
    ax.axvspan(0, 100, alpha=0.15, color='red')
    ax.set_xlabel('time (s)'); ax.set_ylabel('body direction (unwrapped, rad)')
    ax.set_title('Body direction (unwrap): drift = directed motion')

    fig.suptitle(f'{SESSION}: behavioral inspection', fontsize=13)
    fig.tight_layout()
    out = Path(__file__).parent / 'results' / 'window_rqa_clustering' / f'inspect_{SESSION}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
