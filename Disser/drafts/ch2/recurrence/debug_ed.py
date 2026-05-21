#!/usr/bin/env python
"""Diagnose ED-speed correlation: MP-correction × Gaussian smoothing on one session."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d

DRIADA = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA / 'src'))
sys.path.insert(0, str(DRIADA / 'tools'))
from load_synchronized_experiments import load_experiment_from_npz
from driada.dimensionality.effective import eff_dim

NOF = DRIADA / 'DRIADA data/NOF/SynchronizedData26_v1'
SESSION = 'NOF_H06_1D'
DS = 5
FPS_EFF = 4.0
WIN_SEC = 25.0


def run_one(calcium, speed_ds, label):
    win = int(WIN_SEC * FPS_EFF)
    step = win // 2
    n_frames = calcium.shape[1]
    n_win = (n_frames - win) // step + 1
    ed_c = np.zeros(n_win)
    ed_nc = np.zeros(n_win)
    sp = np.zeros(n_win)
    for w in range(n_win):
        i0 = w * step; i1 = i0 + win
        d = calcium[:, i0:i1].T
        try:
            ed_c[w] = eff_dim(d, enable_correction=True)
        except Exception:
            ed_c[w] = np.nan
        ed_nc[w] = eff_dim(d, enable_correction=False)
        sp[w] = speed_ds[i0:i1].mean()
    rc, pc = stats.pearsonr(ed_c[~np.isnan(ed_c)], sp[~np.isnan(ed_c)])
    rn, pn = stats.pearsonr(ed_nc, sp)
    print(f'{label:<30s}  ED_MP-corrected:   r={rc:+.3f}, p={pc:.2e}, '
          f'mean ED={np.nanmean(ed_c):.2f}')
    print(f'{label:<30s}  ED_no-correction:  r={rn:+.3f}, p={pn:.2e}, '
          f'mean ED={ed_nc.mean():.2f}')
    return ed_c, ed_nc, sp


exp = load_experiment_from_npz(NOF / f'{SESSION}_aligned.npz', verbose=False)
N, T_full = exp.calcium.data.shape
print(f'Session: {SESSION}, N={N} neurons, T_full={T_full} frames')
print(f'Window: {int(WIN_SEC*FPS_EFF)} samples = {WIN_SEC}s')
print()

# Variant 1: full-rate, no smoothing
ca = exp.calcium.data
sp = exp.dynamic_features['speed'].data
print('Var 1: raw dF/F at 20Hz, no smoothing')
ca1 = ca; sp1 = sp
win = int(WIN_SEC * 20.0); step = win // 2
n_frames = ca1.shape[1]; n_win = (n_frames - win) // step + 1
ed_c = np.zeros(n_win); ed_nc = np.zeros(n_win); sp_w = np.zeros(n_win)
for w in range(n_win):
    i0 = w*step; i1 = i0+win
    d = ca1[:, i0:i1].T
    try: ed_c[w] = eff_dim(d, enable_correction=True)
    except: ed_c[w] = np.nan
    ed_nc[w] = eff_dim(d, enable_correction=False)
    sp_w[w] = sp1[i0:i1].mean()
mask = ~np.isnan(ed_c)
print(f'  ED_MP   r={stats.pearsonr(ed_c[mask], sp_w[mask])[0]:+.3f}, '
      f'mean={np.nanmean(ed_c):.2f}')
print(f'  ED_raw  r={stats.pearsonr(ed_nc, sp_w)[0]:+.3f}, mean={ed_nc.mean():.2f}')
print()

# Variant 2: downsampled, no smoothing
print('Var 2: downsampled ds=5 (4Hz), no smoothing')
ca2 = ca[:, ::DS]; sp2 = sp[::DS]
run_one(ca2, sp2, '  ')
print()

# Variant 3: downsampled + gaussian smoothing
print('Var 3: downsampled + gaussian(sigma=2)')
ca3 = np.array([gaussian_filter1d(c, 2) for c in ca[:, ::DS]])
run_one(ca3, sp[::DS], '  ')
