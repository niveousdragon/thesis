#!/usr/bin/env python
"""Драфт фигур секции «вовлечённость в популяционный код» (корректная версия).
AE — основной; PCA/UMAP — sanity check. Латенты из cache_v3, декодирование — space_coding_compare.
ВАЖНО: статистика на уровне МЫШИ (n=16), т.к. 4 дня/мышь не независимы (псевдорепликация).
F1: A геометрия (2D-колормап положения) + B декодирование (vs PCA/UMAP/случайно).
F2: A селективные vs нет, B связь с поведением+тренд, C распределение вовлечённости.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, wilcoxon

DRIADA = Path(r"C:\Users\User\PycharmProjects\driada")
sys.path.insert(0, str(DRIADA / "src")); sys.path.insert(0, str(DRIADA / "tools"))
from load_synchronized_experiments import load_experiment_from_npz

HERE = Path(__file__).parent
CACHE = HERE / "cache_v3"
SYNC = DRIADA / "DRIADA data" / "NOF" / "SynchronizedData26_v1"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
DS = 5
EX = "NOF_H01_1D"
# единая палитра кодов (1B и 2C одинаковы): AE зелёный, PCA синий, UMAP серый
cAE, cPCA, cUM, cCH = "#2CA02C", "#2166AC", "#888888", "#cccccc"
cSEL, cSCAT = "#D81B9A", "#2166AC"          # 2A magenta, 2B синий
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def sigbracket(ax, x1, x2, y, p, h):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="k", lw=1.1)
    ax.text((x1 + x2) / 2, y + h, stars(p), ha="center", va="bottom", fontsize=11)


def by_mouse(sessions, vals):
    """dict мышь -> среднее по её дням (только конечные значения)."""
    g = defaultdict(list)
    for s, v in zip(sessions, vals):
        if np.isfinite(v):
            g[str(s).split("_")[1]].append(v)
    return {m: float(np.mean(x)) for m, x in g.items() if x}


def paired_mouse(sessions, a, b):
    da, db = by_mouse(sessions, a), by_mouse(sessions, b)
    mice = sorted(set(da) & set(db))
    return wilcoxon([da[m] for m in mice], [db[m] for m in mice])[1], len(mice)


def pos_for(session):
    exp = load_experiment_from_npz(SYNC / f"{session}_aligned.npz", verbose=False)
    x = exp.dynamic_features["x"].data; y = exp.dynamic_features["y"].data
    vi = np.where(np.isfinite(x) & np.isfinite(y))[0][::DS]
    P = np.column_stack([x[vi], y[vi]]).astype(float)
    return P * (44.0 if np.nanmax(np.abs(P)) < 2 else 1.0)


# ---------- сбор метрик из cache_v3 ----------
files = sorted(glob.glob(str(CACHE / "NOF_*.npz")))
spear = {k: [] for k in ("ae", "umap", "pca")}
sess_inv, sel_v, non_v, rho_v, inv_all, bmi_all = [], [], [], [], [], []
for f in files:
    d = np.load(f, allow_pickle=True); sess = str(d["session"])
    for k in ("ae", "umap", "pca"):
        spear[k].append(d[f"spear_{k}"])
    bs = d["beh_sel"]; sa = d["spear_ae"]
    sess_inv.append(sess)
    sel_v.append(sa[bs].mean() if bs.any() else np.nan)
    non_v.append(sa[~bs].mean() if (~bs).any() else np.nan)
    r, _ = spearmanr(sa, d["beh_mi"]); rho_v.append(r if np.isfinite(r) else np.nan)
    inv_all.append(sa); bmi_all.append(d["beh_mi"])
alls = {k: np.concatenate(v) for k, v in spear.items()}
inv_all = np.concatenate(inv_all); bmi_all = np.concatenate(bmi_all)
rho = float(np.nanmean(rho_v))

# ================= Фигура 1: геометрия + декодирование =================
d_ex = np.load(CACHE / f"{EX}.npz", allow_pickle=True)
ex_Z = d_ex["Z_ae"]; ex_P = pos_for(EX)
c2 = PCA(2, random_state=0).fit_transform(ex_Z)
sc = np.load(HERE / "space_coding_compare.npz", allow_pickle=True)
ses = sc["sessions"]; keys = ["AE", "UMAP", "PCA", "chance"]
mvals = {k: np.array(list(by_mouse(ses, sc[k] * 44).values())) for k in keys}  # 16 мышей
dmean = {k: mvals[k].mean() for k in keys}
dci = {k: 1.96 * mvals[k].std(ddof=1) / np.sqrt(len(mvals[k])) for k in keys}


def pos2color(P):
    xn = (P[:, 0] - P[:, 0].min()) / (np.ptp(P[:, 0]) + 1e-9)
    yn = (P[:, 1] - P[:, 1].min()) / (np.ptp(P[:, 1]) + 1e-9)
    return np.column_stack([xn, yn, 0.6 * np.ones_like(xn)])


fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
ax[0].scatter(c2[:, 0], c2[:, 1], c=pos2color(ex_P), s=7)
ax[0].set_title("A", loc="left", fontweight="bold", fontsize=14)
ax[0].set_xlabel("AE comp 1"); ax[0].set_ylabel("AE comp 2")
gx, gy = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
ins = ax[0].inset_axes([0.76, 0.76, 0.22, 0.22])
ins.imshow(np.dstack([gx, gy, 0.6 * np.ones_like(gx)]), origin="lower", extent=[0, 1, 0, 1])
ins.set_xticks([]); ins.set_yticks([]); ins.set_xlabel("X", fontsize=8); ins.set_ylabel("Y", fontsize=8)
ax[1].bar(["AE", "UMAP", "PCA", "случайно"], [dmean[k] for k in keys],
          yerr=[dci[k] for k in keys], color=[cAE, cUM, cPCA, cCH],
          capsize=4, error_kw={"ecolor": "#333", "lw": 1.2})
ax[1].set_ylabel("ошибка декодирования положения, см")
ax[1].set_title("B", loc="left", fontweight="bold", fontsize=14)
p_au, _ = paired_mouse(ses, sc["AE"], sc["UMAP"])
p_ap, _ = paired_mouse(ses, sc["AE"], sc["PCA"])
p_ac, nm = paired_mouse(ses, sc["AE"], sc["chance"])
top = max(dmean[k] + dci[k] for k in keys)
sigbracket(ax[1], 0, 1, top * 1.02, p_au, top * 0.035)
sigbracket(ax[1], 0, 2, top * 1.13, p_ap, top * 0.035)
sigbracket(ax[1], 0, 3, top * 1.24, p_ac, top * 0.035)
ax[1].set_ylim(0, top * 1.42)
fig.tight_layout(); fig.savefig(OUT / "fig_v3_F1_geometry.png", dpi=190); plt.close(fig)
print(f"saved F1; mice n={nm}; p AE-UMAP {p_au:.1e} AE-PCA {p_ap:.1e} AE-chance {p_ac:.1e}")

# ================= Фигура 2: вовлечённость + поведение =================
sm = by_mouse(sess_inv, sel_v); nmd = by_mouse(sess_inv, non_v)
mice = sorted(set(sm) & set(nmd))
sel_m = np.array([sm[m] for m in mice]); non_m = np.array([nmd[m] for m in mice])
_, pw = wilcoxon(sel_m, non_m)
rho_m = np.array(list(by_mouse(sess_inv, rho_v).values()))
_, p_trend = wilcoxon(rho_m)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
# A
ax[0].bar(["селективные", "не сел."], [sel_m.mean(), non_m.mean()],
          yerr=[1.96 * sel_m.std(ddof=1) / np.sqrt(len(sel_m)),
                1.96 * non_m.std(ddof=1) / np.sqrt(len(non_m))],
          color=[cSEL, cCH], capsize=4)
ax[0].set_ylabel("вовлечённость (Spearman, AE)")
yb = max(sel_m.mean(), non_m.mean()) * 1.18
sigbracket(ax[0], 0, 1, yb, pw, yb * 0.04)
ax[0].set_ylim(0, yb * 1.18)
ax[0].set_title("A", loc="left", fontweight="bold", fontsize=14)
# B
mask = bmi_all > 0; xb, yv = bmi_all[mask], inv_all[mask]
ax[1].scatter(xb, yv, s=4, alpha=0.22, color=cSCAT)
b1, b0 = np.polyfit(xb, yv, 1)
xhi = np.percentile(xb, 99); xx = np.linspace(xb.min(), xhi, 50)
ax[1].plot(xx, b1 * xx + b0, color="k", lw=2)
ax[1].set_xlim(0, xhi)
ax[1].text(0.04, 0.95, f"ρ = {rho:+.2f}\np = {p_trend:.1e} {stars(p_trend)}",
           transform=ax[1].transAxes, va="top", fontsize=10)
ax[1].set_xlabel("поведенческая ВИ нейрона"); ax[1].set_ylabel("вовлечённость (Spearman, AE)")
ax[1].set_title("B", loc="left", fontweight="bold", fontsize=14)
# C
bins = np.linspace(-0.3, 0.7, 40)
ax[2].hist(alls["ae"], bins=bins, color=cAE, alpha=0.8, label="AE")
ax[2].hist(alls["pca"], bins=bins, histtype="step", color=cPCA, lw=1.6, label="PCA")
ax[2].hist(alls["umap"], bins=bins, histtype="step", color=cUM, lw=1.6, label="UMAP")
ax[2].axvline(0, color="k", lw=0.8, ls="--")
ax[2].set_xlabel("вовлечённость (Spearman)"); ax[2].set_ylabel("число нейронов")
ax[2].legend(fontsize=9); ax[2].set_title("C", loc="left", fontweight="bold", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "fig_v3_involvement.png", dpi=190); plt.close(fig)
print(f"saved F2; mice n={len(mice)}; p(sel/non)={pw:.1e} rho={rho:+.2f} p_trend={p_trend:.1e}")
