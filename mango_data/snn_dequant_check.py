"""Dequantization-corrected H_gauss vs raw H_gauss on the 5 main-figure
sublayers across 26 training epochs.

Reviewer concern (W2/B3/C6b): differential entropy is not a strict upper
bound on discrete Shannon entropy without a uniform-noise correction.
Standard fix: shift covariance by I/12 (variance of U[0,1] noise).

This script computes:
  H_gauss_raw     = 0.5 * sum log2(2*pi*e * lambda_k)            (current code)
  H_gauss_dequant = 0.5 * sum log2(2*pi*e * lambda_k_dequant)    (cov + I/12)

over both .sn1 and .sn2 variants of the 5 main-figure blocks
(L2.1, L3.0, L3.1, L4.0, L4.1) at all 26 training epochs.

Output:
  results/dequant_check_per_cell.csv         per (sublayer, epoch) row
  results/dequant_check_summary.txt           text summary + reversal check
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent
SNN_DIR = DATA_ROOT / 'Activity' / 'SJ-SNN-50'
RES_DIR = DATA_ROOT / 'results'
RES_DIR.mkdir(exist_ok=True)

EPOCHS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175,
          200, 225, 250, 275, 300, 400, 500, 600, 700, 800, 900]

# Both .sn1 and .sn2 variants of the 5 main-figure blocks
SUBLAYERS = [
    'layer2.1.sn1', 'layer2.1.sn2',
    'layer3.0.sn1', 'layer3.0.sn2',
    'layer3.1.sn1', 'layer3.1.sn2',
    'layer4.0.sn1', 'layer4.0.sn2',
    'layer4.1.sn1', 'layer4.1.sn2',
]


def load_activity(epoch, layer, split='test'):
    """Replicate load_activity from 07_latent_mi.py."""
    epoch_dir = SNN_DIR / f'iter {epoch}' / split
    if layer == 'sn1':
        fname = f'SJ-SNN act iter {epoch} {split} layer sn1.npy.npz'
    else:
        fname = f'SJ-SNN act iter {epoch} {split} layer {layer}.npy.npz'
    path = epoch_dir / fname
    if not path.exists():
        return None
    data = np.load(path)
    key = 'arr_0' if 'arr_0' in data else 'a'
    arr = data[key]
    return arr.sum(axis=0)  # (n_images, N)


def H_gauss_raw(act):
    """Current code: rank-truncate eigenvalues > 1e-10, sum log2(2*pi*e*lambda)."""
    cov = np.cov(act, rowvar=False)
    eigs = np.linalg.eigvalsh(cov)
    eigs = eigs[eigs > 1e-10]
    return 0.5 * np.sum(np.log2(2 * np.pi * np.e * eigs)), len(eigs)


def H_gauss_dequant(act):
    """Dequantization correction: cov + I/12, then same procedure.

    Variance of U[0, 1] continuous noise = 1/12. Adding to discrete spike
    counts gives a continuous proxy whose differential entropy is a valid
    upper bound on the original discrete Shannon entropy.
    """
    cov = np.cov(act, rowvar=False)
    cov_dq = cov + np.eye(cov.shape[0]) / 12.0
    eigs = np.linalg.eigvalsh(cov_dq)
    eigs = eigs[eigs > 1e-10]
    return 0.5 * np.sum(np.log2(2 * np.pi * np.e * eigs)), len(eigs)


def check_two_phase_reversal(df, sublayer):
    """For a deep sublayer, check whether trajectory peaks at intermediate
    epoch and reverses leftward. Returns (peak_epoch, peak_value, end_value,
    drop) for both raw and dequant.
    """
    sub = df[df.sublayer == sublayer].sort_values('epoch')
    out = {}
    for col in ('H_raw', 'H_dequant'):
        vals = sub[col].values
        if np.isnan(vals).all():
            out[col] = dict(peak_epoch=None, peak=np.nan, end=np.nan, drop=np.nan)
            continue
        peak_idx = int(np.nanargmax(vals))
        peak_epoch = int(sub.iloc[peak_idx].epoch)
        peak_val = float(vals[peak_idx])
        end_val = float(vals[-1])
        drop = peak_val - end_val
        out[col] = dict(peak_epoch=peak_epoch, peak=peak_val,
                        end=end_val, drop=drop)
    return out


def main():
    rows = []
    t0 = time.time()
    for li, layer in enumerate(SUBLAYERS):
        for ei, epoch in enumerate(EPOCHS):
            act = load_activity(epoch, layer)
            if act is None:
                rows.append(dict(sublayer=layer, epoch=epoch,
                                  N_neurons=np.nan, n_samples=np.nan,
                                  H_raw=np.nan, H_dequant=np.nan,
                                  delta=np.nan, n_eigs_kept_raw=np.nan,
                                  n_eigs_kept_dequant=np.nan,
                                  note='missing'))
                continue
            n_samples, n_neurons = act.shape
            try:
                hr, nr = H_gauss_raw(act)
                hd, nd = H_gauss_dequant(act)
                rows.append(dict(sublayer=layer, epoch=epoch,
                                  N_neurons=n_neurons, n_samples=n_samples,
                                  H_raw=hr, H_dequant=hd, delta=hd - hr,
                                  n_eigs_kept_raw=nr,
                                  n_eigs_kept_dequant=nd, note=''))
            except Exception as e:
                rows.append(dict(sublayer=layer, epoch=epoch,
                                  N_neurons=n_neurons, n_samples=n_samples,
                                  H_raw=np.nan, H_dequant=np.nan,
                                  delta=np.nan, n_eigs_kept_raw=np.nan,
                                  n_eigs_kept_dequant=np.nan,
                                  note=f"err:{type(e).__name__}"))
        elapsed = time.time() - t0
        print(f"  [{li+1}/{len(SUBLAYERS)}] {layer} done  ({elapsed:.0f}s)",
              flush=True)

    df = pd.DataFrame(rows)
    out_csv = RES_DIR / 'dequant_check_per_cell.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-cell: {out_csv}")

    # Summary
    lines = []
    lines.append("=" * 78)
    lines.append("Dequantization correction (cov += I/12) on H_gauss")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Per-sublayer summary across 26 epochs:")
    lines.append("")
    hdr = (f"{'sublayer':<16s}{'N':>5s}  "
           f"{'mean H_raw':>11s}{'mean H_deq':>11s}{'mean delta':>11s}  "
           f"{'min delta':>10s}{'max delta':>10s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for layer in SUBLAYERS:
        sub = df[df.sublayer == layer].dropna(subset=['H_raw'])
        if len(sub) == 0:
            continue
        N = int(sub.N_neurons.iloc[0])
        lines.append(
            f"{layer:<16s}{N:>5d}  "
            f"{sub.H_raw.mean():>11.2f}{sub.H_dequant.mean():>11.2f}"
            f"{sub.delta.mean():>11.4f}  "
            f"{sub.delta.min():>10.4f}{sub.delta.max():>10.4f}")
    lines.append("")
    lines.append("Two-phase reversal check (peak vs end-of-training):")
    lines.append("")
    for layer in SUBLAYERS:
        chk = check_two_phase_reversal(df, layer)
        lines.append(f"  {layer}:")
        for variant, label in (('H_raw', 'raw      '),
                                ('H_dequant', 'dequant  ')):
            c = chk[variant]
            if c['peak_epoch'] is None:
                lines.append(f"    {label}: no data")
                continue
            lines.append(
                f"    {label}: peak={c['peak']:.2f} at e={c['peak_epoch']:>3d}, "
                f"end={c['end']:.2f}, drop={c['drop']:.2f} bits")

    lines.append("")
    lines.append("Question: does the leftward reversal in deep sublayers (L4.x)")
    lines.append("survive dequantization?")
    lines.append("")
    for layer in SUBLAYERS:
        if not layer.startswith('layer4'):
            continue
        chk = check_two_phase_reversal(df, layer)
        raw_drops = chk['H_raw']['drop']
        dq_drops = chk['H_dequant']['drop']
        if raw_drops is None or np.isnan(raw_drops):
            continue
        verdict = ("YES" if (raw_drops > 0 and dq_drops > 0
                              and abs(dq_drops - raw_drops) / max(abs(raw_drops), 1e-9) < 0.5)
                   else "PARTIAL" if (raw_drops > 0 and dq_drops > 0)
                   else "NO")
        lines.append(
            f"  {layer}: raw drop {raw_drops:+.2f} bits, "
            f"dequant drop {dq_drops:+.2f} bits  -> reversal preserved: {verdict}")

    summary = "\n".join(lines)
    print()
    print(summary)
    with open(RES_DIR / 'dequant_check_summary.txt', 'w') as f:
        f.write(summary + "\n")
    print(f"\nSaved summary: {RES_DIR / 'dequant_check_summary.txt'}")


if __name__ == "__main__":
    main()
