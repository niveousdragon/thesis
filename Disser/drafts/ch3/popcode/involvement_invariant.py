#!/usr/bin/env python
"""Инвариантная вовлечённость нейронов в популяционный код + контроль на вращение.

Заменяет хрупкую покомпонентную «селективность к оси» на базис-инвариантные меры:
  R2_n  = доля дисперсии нейрона, восстанавливаемая из 16-мерного кода (CV, линейно)
          инвариантна к обратимому линейному репараметру Z (зависит от span, не от осей)
Сравниваем коды: AE (lambda=0) vs UMAP-16 vs PCA-16.
Пересечение с INTENSE: R2_n у поведенчески-селективных vs нет; корр(R2_n, повед. MI).
Контроль на вращение: случайная Q (16x16) -> совместный R2_n не меняется,
  покомпонентные доли (маргинальный R2 по осям) перетасовываются.

Usage: python involvement_invariant.py --session NOF_H01_1D
       python involvement_invariant.py --all
       python involvement_invariant.py --aggregate
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, argparse, warnings, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_popcode_ae as base   # build_exp, AE_DIR, NOF_DATA, FPS_DS, params
from driada.information.info_base import MultiTimeSeries
from driada.intense.pipelines import compute_cell_feat_significance
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import special_ortho_group, spearmanr, wilcoxon

CACHE = HERE / "cache_inv"
DIM = 16
SEED = 42


def cv_r2(X, Yall):
    """X (T,d), Yall (T,N) -> R2 на нейрон (5-fold CV, многовыходная регрессия)."""
    pred = cross_val_predict(LinearRegression(), X, Yall, cv=KFold(5, shuffle=False))
    ss_res = ((Yall - pred) ** 2).sum(0)
    ss_tot = ((Yall - Yall.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def umap16(ca):
    mts = MultiTimeSeries(ca, discrete=False)
    np.random.seed(SEED)
    return mts.get_embedding(method="umap", dim=DIM, min_dist=0.8,
                             n_neighbors=30).coords.T


def run_one(session, force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{session}.npz"
    if out.exists() and not force:
        print(f"  {session}: cached"); return True
    _, mouse, day = session.split("_")
    z = np.load(base.AE_DIR / f"pilot_matched_{mouse}_{day}.npz")
    Z = np.vstack([z["Z_tr"], z["Z_te"]]).astype(np.float64)        # (T,16)
    valid_idx = z["valid_idx"].astype(int)
    matched_idx = z["matched_idx"].astype(int)

    sess = np.load(base.NOF_DATA / f"NOF_{mouse}_{day} syn data.npz")
    ca = sess["calcium"][matched_idx, :][:, valid_idx]              # (N,T)
    Yall = ca.T.astype(np.float64)                                  # (T,N)
    N = Yall.shape[1]
    t0 = time.time()

    # --- градуированная вовлечённость R2_n для трёх кодов ---
    r2_ae = cv_r2(Z, Yall)
    r2_umap = cv_r2(umap16(ca), Yall)
    r2_pca = cv_r2(PCA(DIM, random_state=SEED).fit_transform(ca.T), Yall)

    # null-порог: циклический сдвиг нейронов -> R2 случайности (5 проходов)
    null = []
    for k in range(5):
        rng = np.random.default_rng(SEED + k)
        Ysh = np.column_stack([np.roll(Yall[:, i], int(rng.integers(1, len(Yall) - 1)))
                               for i in range(N)])
        null.append(cv_r2(Z, Ysh))
    thr = np.percentile(np.concatenate(null), 95)
    involved = r2_ae > thr

    # --- поведенческая селективность (INTENSE) ---
    exp = base.build_exp(mouse, day, valid_idx, matched_idx)
    b_stats, b_sig, *_ = compute_cell_feat_significance(
        exp, feat_bunch=None, data_type="calcium", mode="two_stage",
        n_shuffles_stage1=100, n_shuffles_stage2=1000, pval_thr=0.05,
        multicomp_correction="fdr_bh", use_circular_2d=False,
        find_optimal_delays=True, n_jobs=-1, seed=SEED,
        save_computed_stats=False, verbose=False)
    beh_sel = np.zeros(N, dtype=bool)
    beh_relmi = np.zeros(N)            # макс rel_me_beh по значимым фичам
    for nid in b_sig:
        i = int(nid)
        sig = [f for f in b_sig[nid] if b_sig[nid][f].get("stage2", False)]
        if sig:
            beh_sel[i] = True
            beh_relmi[i] = max(b_stats[nid][f].get("rel_me_beh", 0.0) for f in sig)

    # --- контроль на вращение ---
    Q = special_ortho_group.rvs(DIM, random_state=SEED)
    Zrot = Z @ Q
    r2_joint_rot = cv_r2(Zrot, Yall)            # должен совпасть с r2_ae
    joint_invariance = float(np.max(np.abs(r2_ae - r2_joint_rot)))

    def axis_fracs(M):
        return np.sort([(cv_r2(M[:, [e]], Yall) > thr).mean()
                        for e in range(DIM)])[::-1]
    fr_before = axis_fracs(Z)
    fr_after = axis_fracs(Zrot)

    np.savez(out, session=session, N=N, thr=thr,
             r2_ae=r2_ae, r2_umap=r2_umap, r2_pca=r2_pca,
             involved=involved, beh_sel=beh_sel, beh_relmi=beh_relmi,
             joint_invariance=joint_invariance,
             axis_frac_before=fr_before, axis_frac_after=fr_after)
    r_sel = r2_ae[beh_sel].mean() if beh_sel.any() else np.nan
    r_non = r2_ae[~beh_sel].mean() if (~beh_sel).any() else np.nan
    print(f"  {session}: N={N} involved={involved.mean():.2f} "
          f"R2med(AE/UM/PCA)={np.median(r2_ae):.2f}/{np.median(r2_umap):.2f}/"
          f"{np.median(r2_pca):.2f} R2[sel/non]={r_sel:.2f}/{r_non:.2f} "
          f"rotInv={joint_invariance:.1e} ({time.time()-t0:.0f}s)")
    return True


def run_all(force=False):
    sessions = base.list_sessions()
    print(f"=== invariant involvement: {len(sessions)} sessions ===")
    t = time.time(); ok = fail = 0
    for i, s in enumerate(sessions, 1):
        print(f"\n[{i}/{len(sessions)}] {s}")
        try:
            ok += run_one(s, force=force)
        except Exception as e:
            import traceback; print(f"  {s}: FAIL {type(e).__name__}: {e}")
            traceback.print_exc(); fail += 1
    print(f"\n=== done: {ok} ok, {fail} failed, {(time.time()-t)/60:.1f} min ===")


def aggregate():
    files = sorted(CACHE.glob("NOF_*.npz"))
    if not files:
        print("no caches"); return
    r2 = {k: [] for k in ("ae", "umap", "pca")}
    inv_frac, r_sel, r_non, corrs = [], [], [], []
    rot_inv, fr_b, fr_a = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in r2: r2[k].append(d[f"r2_{k}"])
        inv_frac.append(float(d["involved"].mean()))
        bs = d["beh_sel"]; r2ae = d["r2_ae"]
        if bs.any(): r_sel.append(r2ae[bs].mean())
        if (~bs).any(): r_non.append(r2ae[~bs].mean())
        rho, _ = spearmanr(r2ae, d["beh_relmi"])
        if np.isfinite(rho): corrs.append(rho)
        rot_inv.append(float(d["joint_invariance"]))
        fr_b.append(d["axis_frac_before"]); fr_a.append(d["axis_frac_after"])
    allr2 = {k: np.concatenate(v) for k, v in r2.items()}
    med_ae = np.array([np.median(x) for x in r2["ae"]])
    med_um = np.array([np.median(x) for x in r2["umap"]])
    print(f"\n=== AGGREGATE invariant involvement ({len(files)} sessions) ===")
    print("Градуированная вовлечённость R2_n (медиана по нейронам, пул):")
    for k in ("ae", "umap", "pca"):
        print(f"  {k.upper():5s}: median {np.median(allr2[k]):.3f}  "
              f"frac(R2>0.2) {np.mean(allr2[k] > 0.2):.2f}")
    st, pv = wilcoxon(med_ae, med_um)
    print(f"  AE>UMAP по медиане R2 в {int((med_ae>med_um).sum())}/{len(med_ae)} "
          f"сессий (Wilcoxon p={pv:.1e})")
    print(f"\nДоля 'значимо вовлечённых' (R2>null95): {np.mean(inv_frac)*100:.0f}%")
    print(f"R2_n у поведенч.-селективных vs нет: "
          f"{np.mean(r_sel):.3f} vs {np.mean(r_non):.3f}")
    print(f"corr(R2_n, поведенч. rel-MI): rho={np.mean(corrs):.2f} "
          f"(медиана {np.median(corrs):.2f})")
    print(f"\n--- контроль на вращение ---")
    print(f"совместный R2_n инвариантен: max|R2-R2_rot| = {np.mean(rot_inv):.2e} (~0)")
    fr_b, fr_a = np.array(fr_b), np.array(fr_a)
    print(f"покомпонентные доли (топ-ось) до/после вращения: "
          f"{fr_b[:,0].mean():.3f} -> {fr_a[:,0].mean():.3f}; "
          f"разброс по осям (std) {fr_b.std(1).mean():.3f} -> {fr_a.std(1).mean():.3f}")
    np.savez(HERE / "involvement_invariant_summary.npz",
             r2_ae=allr2["ae"], r2_umap=allr2["umap"], r2_pca=allr2["pca"],
             inv_frac=np.array(inv_frac), r_sel=np.array(r_sel),
             r_non=np.array(r_non), corrs=np.array(corrs),
             fr_before=fr_b, fr_after=fr_a, rot_inv=np.array(rot_inv))
    print("\nsaved involvement_invariant_summary.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session"); p.add_argument("--all", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.session: run_one(a.session, force=a.force)
    if a.all: run_all(force=a.force)
    if a.aggregate: aggregate()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
