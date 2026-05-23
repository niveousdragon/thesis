#!/usr/bin/env python
"""v2: вовлечённость нейронов в популяционный код — ВСЕ нейроны, без matched.

Отличия от v1:
  * AE (lambda=0) ПЕРЕОБУЧАЕТСЯ на всех клетках сессии (не matched-подмножество)
  * всё на одной копии (SynchronizedData26_v1), к ней привязан INTENSE_NOF_v3
  * матрица нейрон×ось из MI (me) + значимость stage2  (rel_mi НЕ используется)
  * нелинейный декодер (KNN) для градуированной вовлечённости R2_n: AE/UMAP/PCA
  * поведенческая селективность из INTENSE_NOF_v3: me + stage2 (без rel_mi)

Usage: python involvement_v2.py --session NOF_H01_1D | --all | --aggregate
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, argparse, warnings, time
from pathlib import Path
import numpy as np

DRIADA = Path(r"C:\Users\User\PycharmProjects\driada")
IB = Path(r"C:\Users\User\PycharmProjects\infobottleneck")
sys.path.insert(0, str(DRIADA / "src"))
sys.path.insert(0, str(DRIADA / "tools"))
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

SYNC = DRIADA / "DRIADA data" / "NOF" / "SynchronizedData26_v1"
INTENSE = DRIADA / "DRIADA data" / "NOF" / "INTENSE_NOF_v3" / "results"
CACHE = Path(__file__).parent / "cache_v2"

DS = 5
DIM, HID, EP, BS, LR = 16, 64, 100, 64, 1e-3
N_SHUF1, N_SHUF2, PVAL, MULTICOMP = 100, 1000, 0.05, "fdr_bh"
SEED = 42
SPATIAL = {"place", "x", "y", "walls", "center", "corners"}


def list_sessions():
    return sorted(p.stem.replace("_aligned", "") for p in SYNC.glob("NOF_*_aligned.npz"))


def cv_r2_knn(X, Yall, k=10):
    pred = cross_val_predict(KNeighborsRegressor(k), X, Yall, cv=KFold(5, shuffle=False))
    ss_res = ((Yall - pred) ** 2).sum(0)
    ss_tot = ((Yall - Yall.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def behavioral(session, n_cells):
    """me + stage2 из INTENSE_NOF_v3 (rel_mi НЕ используется)."""
    rp = INTENSE / f"{session}_results.npz"
    beh_sel = np.zeros(n_cells, bool); beh_mi = np.zeros(n_cells)
    beh_nonspat = np.zeros(n_cells, bool)
    if not rp.exists():
        return beh_sel, beh_mi, beh_nonspat, False
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
    return beh_sel, beh_mi, beh_nonspat, True


def run_one(session, force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{session}.npz"
    if out.exists() and not force:
        print(f"  {session}: cached"); return True
    t0 = time.time()
    exp = load_experiment_from_npz(SYNC / f"{session}_aligned.npz", verbose=False)
    ca_full = exp.calcium.scdata                       # (N, T) z-scaled
    N = ca_full.shape[0]
    x = exp.dynamic_features["x"].data; y = exp.dynamic_features["y"].data
    valid = np.isfinite(x) & np.isfinite(y)
    valid_idx = np.where(valid)[0][::DS]
    ca = ca_full[:, valid_idx]                          # (N, n_valid)
    pos = np.column_stack([x[valid_idx], y[valid_idx]]).astype(np.float32)
    T = ca.T.astype(np.float32)                         # (n_valid, N)
    n_tr = int(0.7 * len(T))

    # --- AE lambda=0 на ВСЕХ клетках ---
    Ttr, Tte = T[:n_tr], T[n_tr:]
    m, s = Ttr.mean(0, keepdims=True), Ttr.std(0, keepdims=True); s[s < 1e-8] = 1
    Ztr, Zte, *_ = train_ae_predict(((Ttr - m) / s), pos[:n_tr], ((Tte - m) / s),
                                    pos[n_tr:], latent_dim=DIM, hidden_dim=HID,
                                    epochs=EP, batch_size=BS, lr=LR,
                                    lambda_predict=0.0, seed=SEED, predict_hidden=32)
    Z = np.vstack([Ztr, Zte]).astype(np.float64)        # (n_valid, 16)

    # --- UMAP/PCA-16 на всех клетках ---
    np.random.seed(SEED)
    Zumap = MultiTimeSeries(ca, discrete=False).get_embedding(
        method="umap", dim=DIM, min_dist=0.8, n_neighbors=30).coords.T
    Zpca = PCA(DIM, random_state=SEED).fit_transform(ca.T)

    # --- нелинейная градуированная вовлечённость R2_n (KNN) ---
    Yall = ca.T.astype(np.float64)
    r2 = {"ae": cv_r2_knn(Z, Yall), "umap": cv_r2_knn(Zumap, Yall),
          "pca": cv_r2_knn(Zpca, Yall)}

    # --- матрица нейрон×ось (AE): me + stage2 ---
    expv = load_exp_from_aligned_data(
        "IABS", {"track": "NOF", "animal_id": session, "session": "v"},
        {"calcium": ca.astype(np.float64),
         "x": pos[:, 0].astype(np.float64), "y": pos[:, 1].astype(np.float64)},
        static_features={"fps": 20.0 / DS}, verbose=False, create_circular_2d=False)
    expv.store_embedding(Z, method_name="ae", data_type="calcium",
                         metadata={"ds": 1, "n_components": DIM})
    r = compute_embedding_selectivity(
        expv, embedding_methods=["ae"], data_type="calcium", mode="two_stage",
        n_shuffles_stage1=N_SHUF1, n_shuffles_stage2=N_SHUF2, pval_thr=PVAL,
        multicomp_correction=MULTICOMP, find_optimal_delays=True, n_jobs=-1,
        seed=SEED, verbose=False)["ae"]
    stats, comp_sel = r["stats"], r["component_selectivity"]
    MI = np.zeros((N, DIM)); SIG = np.zeros((N, DIM), bool)
    for nid in stats:
        i = int(nid)
        for e in range(DIM):
            feat = f"ae_comp{e}"
            if feat in stats[nid]:
                MI[i, e] = stats[nid][feat].get("me", 0.0)
    for e in range(DIM):
        for nid in comp_sel.get(e, []):
            SIG[int(nid), e] = True

    # --- поведение (me+stage2, без rel_mi) ---
    beh_sel, beh_mi, beh_nonspat, ok = behavioral(session, N)

    np.savez(out, session=session, N=N, n_valid=len(valid_idx),
             MI=MI, SIG=SIG,
             r2_ae=r2["ae"], r2_umap=r2["umap"], r2_pca=r2["pca"],
             beh_sel=beh_sel, beh_mi=beh_mi, beh_nonspat=beh_nonspat, beh_ok=ok)
    invu = SIG.any(1).mean()
    print(f"  {session}: N={N} unionSIG={invu:.2f} "
          f"R2med(AE/UM/PCA)={np.median(r2['ae']):.2f}/{np.median(r2['umap']):.2f}/"
          f"{np.median(r2['pca']):.2f} beh={beh_sel.sum()} ({time.time()-t0:.0f}s)")
    return True


def run_all(force=False):
    sess = list_sessions()
    print(f"=== involvement v2 (all cells): {len(sess)} sessions ===")
    t = time.time(); ok = fail = 0
    for i, s in enumerate(sess, 1):
        print(f"\n[{i}/{len(sess)}] {s}")
        try:
            ok += run_one(s, force=force)
        except Exception as e:
            import traceback; print(f"  {s}: FAIL {type(e).__name__}: {e}")
            traceback.print_exc(); fail += 1
    print(f"\n=== done: {ok} ok, {fail} failed, {(time.time()-t)/60:.1f} min ===")


def aggregate():
    from scipy.stats import wilcoxon, spearmanr
    files = sorted(CACHE.glob("NOF_*.npz"))
    if not files:
        print("no caches"); return
    r2 = {k: [] for k in ("ae", "umap", "pca")}
    med = {k: [] for k in ("ae", "umap", "pca")}
    r_sel, r_non, corrs, union = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in r2:
            r2[k].append(d[f"r2_{k}"]); med[k].append(np.median(d[f"r2_{k}"]))
        bs = d["beh_sel"]; ra = d["r2_ae"]
        if bs.any(): r_sel.append(ra[bs].mean())
        if (~bs).any(): r_non.append(ra[~bs].mean())
        rho, _ = spearmanr(ra, d["beh_mi"])
        if np.isfinite(rho): corrs.append(rho)
        union.append(d["SIG"].any(1).mean())
    allr2 = {k: np.concatenate(v) for k, v in r2.items()}
    med = {k: np.array(v) for k, v in med.items()}
    print(f"\n=== AGGREGATE v2 (all cells, {len(files)} sessions) ===")
    print("Градуированная вовлечённость R2_n (KNN), медиана/доля>0.2:")
    for k in ("ae", "umap", "pca"):
        print(f"  {k.upper():5s}: median {np.median(allr2[k]):.3f}  frac>0.2 {np.mean(allr2[k]>0.2):.2f}")
    st, pv = wilcoxon(med["ae"], med["umap"])
    print(f"  AE>UMAP по медиане R2 в {int((med['ae']>med['umap']).sum())}/{len(files)} (p={pv:.1e})")
    print(f"\nДоля нейронов в объединении значимых осей (SIG): {np.mean(union)*100:.0f}%")
    print(f"R2_n у поведенч.-селективных vs нет: {np.mean(r_sel):.3f} vs {np.mean(r_non):.3f}")
    print(f"corr(R2_n, поведенч. MI): rho={np.mean(corrs):.2f} (медиана {np.median(corrs):.2f})")
    np.savez(Path(__file__).parent / "involvement_v2_summary.npz",
             r2_ae=allr2["ae"], r2_umap=allr2["umap"], r2_pca=allr2["pca"],
             med_ae=med["ae"], med_umap=med["umap"], med_pca=med["pca"],
             r_sel=np.array(r_sel), r_non=np.array(r_non), corrs=np.array(corrs),
             union=np.array(union))
    print("\nsaved involvement_v2_summary.npz")


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
