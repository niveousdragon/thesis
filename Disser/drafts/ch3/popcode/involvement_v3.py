#!/usr/bin/env python
"""v3: вовлечённость нейронов в коллективные переменные — КОРРЕЛЯЦИОННАЯ мера.

Замена v2 после deep-analysis (.analysis/analysis_recon_quality_2026_05_23):
  * squared-error R² штрафует амплитуду спайковых транзиентов -> заменён на
    КОРРЕЛЯЦИЮ (Pearson/Spearman) предсказанного и истинного сигнала нейрона
  * CV — blocked KFold(shuffle=False): без автокорреляционной утечки (random течёт)
Всё остальное как v2: все клетки, AE(lambda=0) переобучается, UMAP/PCA-16,
матрица нейрон×ось (me+stage2), поведение из INTENSE_NOF_v3 (me+stage2, без rel_mi).

Usage: python involvement_v3.py --session NOF_H01_1D | --all | --aggregate
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, argparse, warnings, time
from pathlib import Path
import numpy as np

DRIADA = Path(r"C:\Users\User\PycharmProjects\driada")
IB = Path(r"C:\Users\User\PycharmProjects\infobottleneck")
sys.path.insert(0, str(DRIADA / "src")); sys.path.insert(0, str(DRIADA / "tools"))
sys.path.insert(0, str(IB / "src"))

from load_synchronized_experiments import load_experiment_from_npz
from driada.experiment.exp_build import load_exp_from_aligned_data
from driada.information.info_base import MultiTimeSeries
from driada.intense.pipelines import compute_embedding_selectivity
from driada.intense.io import load_results
from ib.ae_predict import train_ae_predict
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import rankdata

SYNC = DRIADA / "DRIADA data" / "NOF" / "SynchronizedData26_v1"
INTENSE = DRIADA / "DRIADA data" / "NOF" / "INTENSE_NOF_v3" / "results"
CACHE = Path(__file__).parent / "cache_v3"
DS = 5
DIM, HID, EP, BS, LR = 16, 64, 100, 64, 1e-3
N_SHUF1, N_SHUF2, PVAL, MULTICOMP = 100, 1000, 0.05, "fdr_bh"
SEED = 42
SPATIAL = {"place", "x", "y", "walls", "center", "corners"}


def list_sessions():
    return sorted(p.stem.replace("_aligned", "") for p in SYNC.glob("NOF_*_aligned.npz"))


def colwise_corr(A, B):
    """Pearson по столбцам двух матриц (T,N) -> (N,)."""
    Az = A - A.mean(0); Bz = B - B.mean(0)
    num = (Az * Bz).sum(0)
    den = np.sqrt((Az ** 2).sum(0) * (Bz ** 2).sum(0)) + 1e-12
    return num / den


def _ranks(a):
    """Векторизованные ранги по столбцам (argsort-argsort; быстрее apply_along_axis)."""
    return a.argsort(0).argsort(0).astype(np.float64)


def corr_involvement(X, Yall):
    """blocked-CV предсказание нейронов из эмбеддинга X (KNN), затем Pearson и
    Spearman по столбцам. Возвращает (pearson(N,), spearman(N,))."""
    pred = cross_val_predict(KNeighborsRegressor(10), X, Yall,
                             cv=KFold(5, shuffle=False))
    pear = colwise_corr(Yall, pred)
    spear = colwise_corr(_ranks(Yall), _ranks(pred))   # векторизованный Spearman
    return np.nan_to_num(pear), np.nan_to_num(spear)


def behavioral(session, n_cells):
    rp = INTENSE / f"{session}_results.npz"
    beh_sel = np.zeros(n_cells, bool); beh_mi = np.zeros(n_cells)
    beh_nonspat = np.zeros(n_cells, bool)
    if not rp.exists():
        return beh_sel, beh_mi, beh_nonspat
    res = load_results(str(rp)); sig, stats = res.significance, res.stats
    for nid in sig:
        i = int(nid)
        if i >= n_cells:
            continue
        s2 = [f for f in sig[nid] if sig[nid][f].get("stage2", False)]
        if s2:
            beh_sel[i] = True
            beh_mi[i] = max(stats[nid][f].get("me", 0.0) for f in s2)
            if any(not any(sp in f for sp in SPATIAL) for f in s2):
                beh_nonspat[i] = True
    return beh_sel, beh_mi, beh_nonspat


def run_one(session, force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{session}.npz"
    if out.exists() and not force:
        print(f"  {session}: cached"); return True
    t0 = time.time()
    exp = load_experiment_from_npz(SYNC / f"{session}_aligned.npz", verbose=False)
    ca_full = exp.calcium.scdata; N = ca_full.shape[0]
    x = exp.dynamic_features["x"].data; y = exp.dynamic_features["y"].data
    valid_idx = np.where(np.isfinite(x) & np.isfinite(y))[0][::DS]
    ca = ca_full[:, valid_idx]; pos = np.column_stack([x[valid_idx], y[valid_idx]]).astype(np.float32)
    Yall = ca.T.astype(np.float64)
    ntr = int(0.7 * len(Yall))

    # AE lambda=0 (все клетки)
    Ttr, Tte = Yall[:ntr], Yall[ntr:]
    m, s = Ttr.mean(0, keepdims=True), Ttr.std(0, keepdims=True); s[s < 1e-8] = 1
    Ztr, Zte, *_ = train_ae_predict(((Ttr - m) / s).astype(np.float32), pos[:ntr],
                                    ((Tte - m) / s).astype(np.float32), pos[ntr:],
                                    latent_dim=DIM, hidden_dim=HID, epochs=EP,
                                    batch_size=BS, lr=LR, lambda_predict=0.0,
                                    seed=SEED, predict_hidden=32)
    Z = np.vstack([Ztr, Zte]).astype(np.float64)
    np.random.seed(SEED)
    Zumap = MultiTimeSeries(ca, discrete=False).get_embedding(
        method="umap", dim=DIM, min_dist=0.8, n_neighbors=30).coords.T
    Zpca = PCA(DIM, random_state=SEED).fit_transform(ca.T)

    # корреляционная вовлечённость (blocked CV)
    pe_ae, sp_ae = corr_involvement(Z, Yall)
    pe_um, sp_um = corr_involvement(Zumap, Yall)
    pe_pc, sp_pc = corr_involvement(Zpca, Yall)

    beh_sel, beh_mi, beh_nonspat = behavioral(session, N)
    # MI/SIG матрица нейрон×ось — переиспользуем из cache_v2 (идентична: тот же seed→тот же AE→тот же INTENSE)
    v2 = Path(__file__).parent / "cache_v2" / f"{session}.npz"
    if v2.exists():
        dv = np.load(v2, allow_pickle=True); MI, SIG = dv["MI"], dv["SIG"]
    else:
        MI = np.zeros((N, DIM)); SIG = np.zeros((N, DIM), bool)

    np.savez(out, session=session, N=N,
             pear_ae=pe_ae, pear_umap=pe_um, pear_pca=pe_pc,
             spear_ae=sp_ae, spear_umap=sp_um, spear_pca=sp_pc,
             Z_ae=Z, Z_umap=Zumap, Z_pca=Zpca,    # СОХРАНЯЕМ тяжёлые латенты (AE дорогой!)
             MI=MI, SIG=SIG, beh_sel=beh_sel, beh_mi=beh_mi, beh_nonspat=beh_nonspat)
    print(f"  {session}: N={N} Pearson med AE/UM/PCA="
          f"{np.median(pe_ae):.2f}/{np.median(pe_um):.2f}/{np.median(pe_pc):.2f} "
          f"frac>0(AE)={np.mean(pe_ae>0):.2f} beh={beh_sel.sum()} ({time.time()-t0:.0f}s)")
    return True


def run_all(force=False):
    sess = list_sessions()
    print(f"=== involvement v3 (correlation, blocked CV): {len(sess)} sessions ===")
    t = time.time(); ok = fail = 0
    for i, s in enumerate(sess, 1):
        print(f"\n[{i}/{len(sess)}] {s}")
        try:
            ok += run_one(s, force=force)
        except Exception as e:
            import traceback; print(f"  {s}: FAIL {type(e).__name__}: {e}"); traceback.print_exc(); fail += 1
    print(f"\n=== done: {ok} ok, {fail} failed, {(time.time()-t)/60:.1f} min ===")


def aggregate():
    from scipy.stats import wilcoxon, spearmanr
    files = sorted(CACHE.glob("NOF_*.npz"))
    if not files:
        print("no caches"); return
    pe = {k: [] for k in ("ae", "umap", "pca")}; med = {k: [] for k in pe}
    p_sel, p_non, corrs, union = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in pe:
            pe[k].append(d[f"pear_{k}"]); med[k].append(np.median(d[f"pear_{k}"]))
        bs = d["beh_sel"]; pa = d["pear_ae"]
        if bs.any(): p_sel.append(pa[bs].mean())
        if (~bs).any(): p_non.append(pa[~bs].mean())
        rho, _ = spearmanr(pa, d["beh_mi"])
        if np.isfinite(rho): corrs.append(rho)
        union.append(d["SIG"].any(1).mean())
    allpe = {k: np.concatenate(v) for k, v in pe.items()}; med = {k: np.array(v) for k, v in med.items()}
    print(f"\n=== AGGREGATE v3 (correlation, {len(files)} sessions) ===")
    print("Корреляционная вовлечённость (Pearson pred~true, blocked CV):")
    for k in ("ae", "umap", "pca"):
        print(f"  {k.upper():5s}: median {np.median(allpe[k]):+.3f}  "
              f"frac>0 {np.mean(allpe[k]>0):.2f}  frac>0.1 {np.mean(allpe[k]>0.1):.2f}")
    st, pv = wilcoxon(med["ae"], med["umap"])
    print(f"  AE>UMAP по медиане Pearson в {int((med['ae']>med['umap']).sum())}/{len(files)} (p={pv:.1e})")
    print(f"\nДоля в объединении значимых осей (MI/stage2): {np.mean(union)*100:.0f}%")
    print(f"Pearson-вовлечённость у поведенч.-селективных vs нет: "
          f"{np.mean(p_sel):+.3f} vs {np.mean(p_non):+.3f}")
    print(f"corr(Pearson-вовлечённость, поведенч. MI): rho={np.mean(corrs):+.2f} "
          f"(медиана {np.median(corrs):+.2f})")
    np.savez(Path(__file__).parent / "involvement_v3_summary.npz",
             pear_ae=allpe["ae"], pear_umap=allpe["umap"], pear_pca=allpe["pca"],
             med_ae=med["ae"], med_umap=med["umap"], p_sel=np.array(p_sel),
             p_non=np.array(p_non), corrs=np.array(corrs), union=np.array(union))
    print("\nsaved involvement_v3_summary.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session"); p.add_argument("--all", action="store_true")
    p.add_argument("--aggregate", action="store_true"); p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.session: run_one(a.session, force=a.force)
    if a.all: run_all(force=a.force)
    if a.aggregate: aggregate()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
