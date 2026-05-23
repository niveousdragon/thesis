#!/usr/bin/env python
"""Вовлечённость нейронов в популяционный код на AE-латентах (infobottleneck).

Вместо UMAP берём готовые AE-эмбеддинги (lambda=0, unsupervised, ds=5, 16-мерные)
из проекта infobottleneck и повторяем на них анализ вовлечённости:
  1. AE-латент Z (matched-клетки) как коллективные переменные
  2. compute_embedding_selectivity (INTENSE): нейрон <-> компонента Z
  3. поведенческая селективность тех же matched-клеток (фрешем, на той же сессии)
  4. перекрытие (полный набор / непространственно / обогащение)

Самодостаточно: Experiment строится из IB-копии NOF (driada/science/NOF data),
matched calcium + поведение выровнены по valid_idx из npz латента.

Usage:
    python run_popcode_ae.py --session NOF_H01_1D
    python run_popcode_ae.py --all
    python run_popcode_ae.py --aggregate
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, time, argparse, warnings
from pathlib import Path
import numpy as np

DRIADA = Path(r"C:\Users\User\PycharmProjects\driada")
sys.path.insert(0, str(DRIADA / "src"))
sys.path.insert(0, str(DRIADA / "tools"))

from driada.experiment.exp_build import load_exp_from_aligned_data
from driada.intense.pipelines import (compute_embedding_selectivity,
                                       compute_cell_feat_significance)

IB = Path(r"C:\Users\User\PycharmProjects\infobottleneck")
AE_DIR = IB / "results" / "nof" / "data_matched_ds5"      # lambda=0, ds=5
NOF_DATA = DRIADA / "science" / "NOF data"
MATCH_DIR = IB / "data" / "NOF" / "Matching"
CACHE = Path(__file__).parent / "cache_ae"

FPS_DS = 4.0          # 20 Гц / ds5
N_SHUF1, N_SHUF2 = 100, 1000
PVAL_THR, MULTICOMP = 0.05, "fdr_bh"
SEED = 42

BEH_KEYS = ["x", "y", "speed", "headdirection", "bodydirection",
            "rest", "walk", "locomotion", "freezing", "rear",
            "corners", "walls", "center",
            "object1", "object2", "object3", "object4", "objects"]
# пространственные фичи (для непространственного перекрытия)
SPATIAL = {"x", "y", "corners", "walls", "center"}


def list_sessions():
    out = []
    for p in sorted(AE_DIR.glob("pilot_matched_*.npz")):
        stem = p.stem.replace("pilot_matched_", "")   # H01_1D
        out.append("NOF_" + stem)                      # NOF_H01_1D
    return out


def build_exp(mouse, day, valid_idx, matched_idx):
    sess = np.load(NOF_DATA / f"NOF_{mouse}_{day} syn data.npz")
    ca = sess["calcium"][matched_idx, :][:, valid_idx]        # (n_matched, n_valid)
    data = {"calcium": ca.astype(np.float64)}
    for k in BEH_KEYS:
        if k not in sess.files:
            continue
        v = sess[k][valid_idx].astype(np.float64)
        if not np.all(np.isfinite(v)) or np.nanstd(v) < 1e-9:
            continue   # пропускаем константные/вырожденные фичи (напр. отсутствующий объект)
        data[k] = v
    exp = load_exp_from_aligned_data(
        data_source="IABS",
        exp_params={"track": "NOF", "animal_id": mouse, "session": day},
        data=data, static_features={"fps": FPS_DS}, verbose=False,
        create_circular_2d=False)
    return exp


def run_one(session, force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{session}.npz"
    if out.exists() and not force:
        print(f"  {session}: cached"); return True
    _, mouse, day = session.split("_")       # NOF, H01, 1D
    aep = AE_DIR / f"pilot_matched_{mouse}_{day}.npz"
    if not aep.exists():
        print(f"  {session}: SKIP (no AE npz)"); return False

    t0 = time.time()
    z = np.load(aep)
    Z = np.vstack([z["Z_tr"], z["Z_te"]]).astype(np.float64)   # (n_valid, 16)
    valid_idx = z["valid_idx"].astype(int)
    matched_idx = z["matched_idx"].astype(int)
    ncomp = Z.shape[1]

    exp = build_exp(mouse, day, valid_idx, matched_idx)
    n_cells = exp.n_cells
    if Z.shape[0] != exp.n_frames:
        print(f"  {session}: WARN Z {Z.shape[0]} != n_frames {exp.n_frames}")

    # --- поведенческая селективность тех же matched-клеток (ДО embedding,
    #     иначе feat_bunch=None подхватит временные ae_comp*) ---
    b_stats, b_sig, *_ = compute_cell_feat_significance(
        exp, feat_bunch=None, data_type="calcium",
        mode="two_stage", n_shuffles_stage1=N_SHUF1, n_shuffles_stage2=N_SHUF2,
        pval_thr=PVAL_THR, multicomp_correction=MULTICOMP, use_circular_2d=False,
        find_optimal_delays=True, n_jobs=-1, seed=SEED,
        save_computed_stats=False, verbose=False)
    beh_all, beh_nonspat = set(), set()
    for nid in b_sig:
        sigfeats = [f for f in b_sig[nid] if b_sig[nid][f].get("stage2", False)]
        if sigfeats:
            beh_all.add(int(nid))
        if any(not any(sp in f for sp in SPATIAL) for f in sigfeats):
            beh_nonspat.add(int(nid))

    # --- AE-латент как embedding ---
    exp.store_embedding(Z, method_name="ae", data_type="calcium",
                        metadata={"ds": 1, "n_components": ncomp})

    emb_res = compute_embedding_selectivity(
        exp, embedding_methods=["ae"], data_type="calcium",
        mode="two_stage", n_shuffles_stage1=N_SHUF1, n_shuffles_stage2=N_SHUF2,
        pval_thr=PVAL_THR, multicomp_correction=MULTICOMP,
        find_optimal_delays=True, n_jobs=-1, seed=SEED, verbose=False)
    r = emb_res["ae"]
    comp_sel = r["component_selectivity"]
    stats = r["stats"]

    n_sig_native = np.array([len(comp_sel.get(c, [])) for c in range(ncomp)])
    order = np.argsort(n_sig_native)[::-1]
    n_sig = np.zeros(ncomp, dtype=int)
    rel_obj = np.empty(ncomp, dtype=object)
    for rank, comp in enumerate(order):
        feat = f"ae_comp{comp}"
        nids = comp_sel.get(comp, [])
        n_sig[rank] = len(nids)
        vals = [stats[nid][feat]["rel_me_beh"] for nid in nids
                if feat in stats.get(nid, {})]
        rel_obj[rank] = np.array(vals, dtype=float)
    emb_selective = set(int(n) for n in r["significant_neurons"].keys())

    def frac(s):
        return len(emb_selective & s) / len(emb_selective) if emb_selective else np.nan
    ov_all = frac(beh_all)
    ov_non = frac(beh_nonspat)
    base = len(beh_all) / max(n_cells, 1)

    np.savez(out, session=session, n_cells=n_cells, n_components=ncomp,
             n_sig=n_sig, frac_sig=n_sig / max(n_cells, 1), rel_mi=rel_obj,
             emb_selective=np.array(sorted(emb_selective)),
             n_emb_sel=len(emb_selective), n_beh_all=len(beh_all),
             n_beh_nonspat=len(beh_nonspat), base_rate=base,
             overlap_all=ov_all, overlap_nonspat=ov_non)
    print(f"  {session}: cells={n_cells} emb-sel={len(emb_selective)} "
          f"beh={len(beh_all)} ov_all={ov_all:.2f} ov_non={ov_non:.2f} "
          f"frac_sig[0]={n_sig[0]/max(n_cells,1):.3f} ({time.time()-t0:.0f}s)")
    return True


def run_all(force=False):
    sessions = list_sessions()
    print(f"=== popcode-AE: {len(sessions)} sessions (lambda=0, ds5, dim=16) ===")
    t = time.time(); ok = fail = 0
    for i, s in enumerate(sessions, 1):
        print(f"\n[{i}/{len(sessions)}] {s}")
        try:
            ok += run_one(s, force=force)
        except Exception as e:
            import traceback; print(f"  {s}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc(); fail += 1
    print(f"\n=== done: {ok} ok, {fail} failed, {(time.time()-t)/60:.1f} min ===")


def aggregate():
    files = sorted(CACHE.glob("NOF_*.npz"))
    if not files:
        print("no caches"); return
    fs, ovA, ovN, base = [], [], [], []
    rel_by, ncomp_ref = None, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        nc = int(d["n_components"])
        if ncomp_ref is None:
            ncomp_ref = nc; rel_by = [[] for _ in range(nc)]
        if nc != ncomp_ref: continue
        fs.append(d["frac_sig"])
        for arr, key in ((ovA, "overlap_all"), (ovN, "overlap_nonspat"),
                         (base, "base_rate")):
            v = float(d[key])
            if not np.isnan(v): arr.append(v)
        rel = d["rel_mi"]
        for c in range(nc):
            rel_by[c].append(np.asarray(rel[c], dtype=float))
    fs = np.array(fs)
    mean_fs = fs.mean(0) * 100
    ci = 1.96 * fs.std(0, ddof=1) / np.sqrt(len(fs)) * 100
    ovA, ovN, base = map(lambda a: np.array(a), (ovA, ovN, base))
    print(f"\n=== AGGREGATE AE-embeddings ({len(files)} sessions) ===")
    print("Доля значимых нейронов на компоненту AE (%, mean+-95%CI):")
    for c in range(ncomp_ref):
        print(f"  ae {c+1:2d}: {mean_fs[c]:.2f} +- {ci[c]:.2f}")
    print(f"\nБаза поведенческой селективности: {base.mean()*100:.0f}% +- {base.std(ddof=1)*100:.0f}%")
    print(f"Перекрытие, полный набор:        {ovA.mean()*100:.0f}% +- {ovA.std(ddof=1)*100:.0f}%")
    print(f"Перекрытие, непространственное:   {ovN.mean()*100:.0f}% +- {ovN.std(ddof=1)*100:.0f}%")
    enr = ovA / base
    print(f"Обогащение над базой: {enr.mean():.2f} +- {enr.std(ddof=1):.2f}")
    rel_obj = np.empty(ncomp_ref, dtype=object)
    for c in range(ncomp_ref):
        rel_obj[c] = np.concatenate(rel_by[c]) if rel_by[c] else np.array([])
    np.savez(Path(__file__).parent / "popcode_ae_summary.npz",
             mean_frac_sig=mean_fs, ci_frac_sig=ci, rel_mi_pooled=rel_obj,
             overlap_all=ovA.mean(), overlap_nonspat=ovN.mean(),
             base_rate=base.mean(), enrichment=enr.mean(), n_sessions=len(files))
    print("\nsaved popcode_ae_summary.npz")


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
