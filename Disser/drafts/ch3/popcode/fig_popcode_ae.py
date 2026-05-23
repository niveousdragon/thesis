#!/usr/bin/env python
"""Фигуры вовлечённости на AE-латентах + сравнение UMAP vs AE.
Читает popcode_ae_summary.npz (и popcode_summary.npz для сравнения)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = Path(__file__).parent
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_fraction(d, tag, label):
    mean_fs, ci = d["mean_frac_sig"], d["ci_frac_sig"]
    n = len(mean_fs); x = np.arange(n)
    colors = plt.cm.viridis(np.linspace(0, 0.92, n))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x, mean_fs, yerr=ci, color=colors, capsize=2,
           error_kw={"ecolor": "#333", "lw": 1})
    ax.set_xticks(x); ax.set_xticklabels([f"{label} {i+1}" for i in range(n)],
                                          rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Коллективная переменная"); ax.set_ylabel("Доля нейронов, %")
    fig.tight_layout(); fig.savefig(OUT / f"fig39_{tag}.png", dpi=200); plt.close(fig)
    print(f"saved fig39_{tag}.png")


def fig_relmi(d, tag, label):
    rel = d["rel_mi_pooled"]; n = len(rel)
    colors = plt.cm.viridis(np.linspace(0, 0.92, n))
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = np.linspace(0, 0.06, 400)
    for c in range(n):
        a = np.asarray(rel[c], dtype=float); a = a[np.isfinite(a)]
        if len(a) < 5: continue
        ax.plot(xs, gaussian_kde(a)(xs), color=colors[c], lw=1.8,
                label=f"{label} {c+1}")
    ax.set_xlabel("Относительная MI"); ax.set_ylabel("Плотность")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / f"fig40_{tag}.png", dpi=200); plt.close(fig)
    print(f"saved fig40_{tag}.png")


def comparison(ae, umap):
    """Сводная панель: доли значимых (нормировано по рангу) + бар перекрытий."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    a = axes[0]
    for d, lab, col in ((umap, "UMAP", "#1f77b4"), (ae, "AE (λ=0)", "#d62728")):
        fs = d["mean_frac_sig"]; r = np.arange(1, len(fs) + 1)
        a.plot(r, fs, "o-", color=col, label=lab)
    a.set_xlabel("Ранг коллективной переменной"); a.set_ylabel("Доля значимых нейронов, %")
    a.legend(); a.set_title("A. Вовлечённость по переменным", loc="left", fontweight="bold")

    b = axes[1]
    cats = ["база\nповедения", "перекрытие\n(всё)", "перекрытие\n(непростр.)", "обогащение\n×"]
    umap_v = [44, 49, 36, 1.11 * 40]    # обогащение масштаб ×40 для видимости
    ae_v = [float(ae["base_rate"]) * 100, float(ae["overlap_all"]) * 100,
            float(ae["overlap_nonspat"]) * 100, float(ae["enrichment"]) * 40]
    xb = np.arange(len(cats)); w = 0.38
    b.bar(xb - w/2, umap_v, w, color="#1f77b4", label="UMAP")
    b.bar(xb + w/2, ae_v, w, color="#d62728", label="AE (λ=0)")
    b.set_xticks(xb); b.set_xticklabels(cats, fontsize=9)
    b.set_ylabel("%  (обогащение ×40)"); b.legend()
    b.set_title("B. Перекрытие с поведением", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig_compare_umap_ae.png", dpi=200); plt.close(fig)
    print("saved fig_compare_umap_ae.png")


def main():
    ae = np.load(HERE / "popcode_ae_summary.npz", allow_pickle=True)
    fig_fraction(ae, "ae_64", "ae"); fig_relmi(ae, "ae_64", "ae")
    up = HERE / "popcode_summary.npz"
    if up.exists():
        comparison(ae, np.load(up, allow_pickle=True))


if __name__ == "__main__":
    main()
