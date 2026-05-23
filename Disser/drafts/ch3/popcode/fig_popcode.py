#!/usr/bin/env python
"""Фигуры вовлечённости в популяционный код по 64 NOF сессиям.

Аналоги рис. 39 (доля значимых нейронов на коллективную переменную) и
рис. 40 (распределения относительной ВИ). Читает popcode_summary.npz.

Usage: python fig_popcode.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = Path(__file__).parent
SUMMARY = HERE / "popcode_summary.npz"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_fraction(d):
    mean_fs = d["mean_frac_sig"]
    ci = d["ci_frac_sig"]
    ncomp = len(mean_fs)
    x = np.arange(ncomp)
    colors = plt.cm.plasma(np.linspace(0, 0.92, ncomp))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x, mean_fs, yerr=ci, color=colors, capsize=3,
           error_kw={"ecolor": "#333333", "lw": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([f"umap {i+1}" for i in range(ncomp)], rotation=0)
    ax.set_xlabel("Коллективная переменная")
    ax.set_ylabel("Доля нейронов, %")
    fig.tight_layout()
    fig.savefig(OUT / "fig39_fraction_significant_64.png", dpi=200)
    plt.close(fig)
    print("saved fig39_fraction_significant_64.png")


def fig_relmi(d):
    rel = d["rel_mi_pooled"]
    ncomp = len(rel)
    colors = plt.cm.plasma(np.linspace(0, 0.92, ncomp))
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = np.linspace(0, max(0.05, max((a.max() if len(a) else 0.05) for a in rel)), 400)
    for c in range(ncomp):
        a = np.asarray(rel[c], dtype=float)
        a = a[np.isfinite(a)]
        if len(a) < 5:
            continue
        kde = gaussian_kde(a)
        ax.plot(xs, kde(xs), color=colors[c], lw=2, label=f"umap {c+1}")
    ax.set_xlabel("Относительная MI")
    ax.set_ylabel("Плотность")
    ax.legend(frameon=True, fontsize=10, ncol=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig40_relmi_distributions_64.png", dpi=200)
    plt.close(fig)
    print("saved fig40_relmi_distributions_64.png")


def main():
    if not SUMMARY.exists():
        print("run: python run_popcode.py --aggregate  (no summary yet)")
        return
    d = np.load(SUMMARY, allow_pickle=True)
    fig_fraction(d)
    fig_relmi(d)


if __name__ == "__main__":
    main()
