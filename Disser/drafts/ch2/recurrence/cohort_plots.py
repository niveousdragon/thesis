#!/usr/bin/env python
"""
Cohort-level figures for window-RQA clustering on NOF (1D sessions).

Reads cohort_summary.npz from both real and shuffled runs, produces:
  Fig A — per-mouse forest plot: speed difference (high-DET − low-DET) ± CI
  Fig B — η² histogram: real vs shuffled
  Fig C — paired η² real vs shuffled (per session)
  Fig D — example trajectory comparison: best session vs H26 (anti-pattern)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).parent
REAL = HERE / 'results' / 'window_rqa_clustering' / 'data' / 'cohort_summary.npz'
SHUF = HERE / 'results' / 'window_rqa_clustering_shuffled' / 'data' / 'cohort_summary.npz'
OUT = HERE / 'results' / 'window_rqa_clustering'


def load_cohort(path):
    d = np.load(path, allow_pickle=True)
    sessions = list(d['sessions'])
    diffs = np.array(d['diffs'])
    eta2s = np.array(d['eta2s'])
    pvals = np.array(d['p_values'])
    return sessions, diffs, eta2s, pvals


def short(s):
    """NOF_H32_1D -> H32, LNOF_J06_1D -> J06."""
    return s.split('_')[1]


def main():
    sR, dR, eR, pR = load_cohort(REAL)
    sS, dS, eS, pS = load_cohort(SHUF)

    # Restrict to NOF only (LNOF were carry-overs from earlier runs)
    mask = np.array([s.startswith('NOF') for s in sR])
    sR = [s for s, m in zip(sR, mask) if m]
    dR, eR, pR = dR[mask], eR[mask], pR[mask]

    mask = np.array([s.startswith('NOF') for s in sS])
    sS = [s for s, m in zip(sS, mask) if m]
    dS, eS, pS = dS[mask], eS[mask], pS[mask]

    # Align sessions between real and shuffled
    common = sorted(set(sR) & set(sS))
    iR = [sR.index(c) for c in common]
    iS = [sS.index(c) for c in common]
    dR, eR, pR = dR[iR], eR[iR], pR[iR]
    dS, eS, pS = dS[iS], eS[iS], pS[iS]
    sessions = common
    labels = [short(s) for s in sessions]
    n = len(sessions)

    # Sort by real diff (most negative = strongest effect)
    order = np.argsort(dR)
    sessions = [sessions[i] for i in order]
    labels = [labels[i] for i in order]
    dR, eR, pR = dR[order], eR[order], pR[order]
    dS, eS, pS = dS[order], eS[order], pS[order]

    # Cohort tests
    sign_real = (dR < 0).sum()
    sign_shuf = (dS < 0).sum()
    binom_real = stats.binomtest(sign_real, n, p=0.5, alternative='greater').pvalue
    binom_shuf = stats.binomtest(sign_shuf, n, p=0.5, alternative='greater').pvalue
    # Wilcoxon: H0 diff distribution symmetric around 0
    w_real = stats.wilcoxon(dR, alternative='less').pvalue
    w_shuf = stats.wilcoxon(dS, alternative='less').pvalue

    print(f'NOF cohort (n={n}):')
    print(f'  Sign test: real {sign_real}/{n} (p={binom_real:.2e}); '
          f'shuf {sign_shuf}/{n} (p={binom_shuf:.2e})')
    print(f'  Wilcoxon (one-sided diff<0): real p={w_real:.2e}, shuf p={w_shuf:.2e}')
    print(f'  Median eta2: real {np.median(eR):.3f}, shuf {np.median(eS):.3f}')

    # =================================================================
    # Combined 4-panel figure
    # =================================================================
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # --- Panel A: per-session forest plot of speed difference ---
    ax = fig.add_subplot(gs[0, :])
    y = np.arange(n)
    c_real = ['#cc4444' if d < 0 else '#888888' for d in dR]
    c_shuf = ['#aaaaaa'] * n
    # offset shuffled slightly below real
    ax.barh(y + 0.18, dR, height=0.36, color=c_real, edgecolor='k',
            linewidth=0.4, label='Real')
    ax.barh(y - 0.18, dS, height=0.36, color=c_shuf, edgecolor='k',
            linewidth=0.4, alpha=0.7, label='Shuffled')
    ax.axvline(0, color='k', lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Speed difference: high-DET − low-DET cluster (cm/s)')
    ax.set_title(
        f'A. Per-session effect (NOF Day-1, n={n}). Negative = high-DET slower (expected).\n'
        f'Real: {sign_real}/{n} negative, sign-test p={binom_real:.1e}, '
        f'Wilcoxon p={w_real:.1e}.  '
        f'Shuffled: {sign_shuf}/{n} negative.', fontsize=11, loc='left')
    # mark significant sessions with *
    for i, p in enumerate(pR):
        if p < 0.05:
            ax.text(min(dR) - 0.4, i + 0.18, '*', ha='center', va='center',
                    fontsize=14, color='red', fontweight='bold')
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    # --- Panel B: eta2 histogram real vs shuffled ---
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, max(eR.max(), eS.max()) + 0.05, 18)
    ax.hist(eS, bins=bins, alpha=0.6, color='gray', edgecolor='k',
            label=f'Shuffled (median {np.median(eS):.3f})')
    ax.hist(eR, bins=bins, alpha=0.6, color='#cc4444', edgecolor='k',
            label=f'Real (median {np.median(eR):.3f})')
    ax.axvline(0.14, color='b', ls=':', alpha=0.5, label='Cohen large (0.14)')
    ax.set_xlabel('η² (window cluster ↔ speed)')
    ax.set_ylabel('Sessions')
    ax.set_title('B. Effect size distribution', fontsize=11, loc='left')
    ax.legend(fontsize=9)

    # --- Panel C: paired eta2 ---
    ax = fig.add_subplot(gs[1, 1])
    for i in range(n):
        ax.plot([0, 1], [eS[i], eR[i]], '-', color='gray', alpha=0.5, lw=0.7)
    ax.scatter([0]*n, eS, s=40, color='gray', edgecolor='k', linewidth=0.5,
               zorder=3, label='Shuffled')
    ax.scatter([1]*n, eR, s=40, color='#cc4444', edgecolor='k', linewidth=0.5,
               zorder=3, label='Real')
    # Highlight H26
    if 'NOF_H26_1D' in sessions:
        i26 = sessions.index('NOF_H26_1D')
        ax.scatter([1], [eR[i26]], s=90, facecolor='none', edgecolor='blue',
                   linewidth=1.8, zorder=4)
        ax.text(1.06, eR[i26], 'H26\n(anti-pattern)', fontsize=8, color='blue',
                va='center')
    # Wilcoxon paired
    w_paired = stats.wilcoxon(eR, eS, alternative='greater').pvalue
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Shuffled', 'Real'])
    ax.set_xlim(-0.3, 1.4)
    ax.set_ylabel('η²')
    ax.set_title(f'C. Paired comparison (Wilcoxon p={w_paired:.1e})', fontsize=11, loc='left')
    ax.legend(fontsize=9, loc='upper left')

    fig.suptitle(
        f'Window-RQA clustering on hippocampal population dynamics ({n} mice, NOF Day-1)',
        fontsize=13, y=0.995)
    out_path = OUT / 'fig_cohort_effect.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

    # =================================================================
    # Save tidy CSV
    # =================================================================
    csv_path = OUT / 'cohort_summary.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('session,mouse,diff_real,eta2_real,p_real,diff_shuf,eta2_shuf,p_shuf\n')
        for i, s in enumerate(sessions):
            f.write(f'{s},{labels[i]},{dR[i]:.4f},{eR[i]:.4f},{pR[i]:.4e},'
                    f'{dS[i]:.4f},{eS[i]:.4f},{pS[i]:.4e}\n')
    print(f'Saved: {csv_path}')


if __name__ == '__main__':
    main()
