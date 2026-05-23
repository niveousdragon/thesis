#!/usr/bin/env python
"""
Вовлечённость отдельных нейронов в популяционный код (NOF), на всех 64 сессиях.

Конвейер на сессию:
  1. UMAP-вложение популяционной активности (dim=10, min_dist=0.8) -> коллективные переменные
  2. compute_embedding_selectivity (INTENSE): нейрон <-> коллективная переменная
     -> доля значимых нейронов на переменную, относительная ВИ (rel_me_beh)
  3. Поведенческая селективность -- из готового прогона INTENSE_NOF_v3
  4. Перекрытие: какая доля нейронов попул. кода также поведенчески селективна

Кэш на сессию: cache/{session}.npz. Агрегация -> popcode_summary.npz + печать чисел.

Usage:
    python run_popcode.py --session NOF_H01_1D   # тест на одной сессии
    python run_popcode.py --all                  # все 64 (пропуск кэшированных)
    python run_popcode.py --aggregate            # свод по кэшам
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

DRIADA_ROOT = Path(r"C:\Users\User\PycharmProjects\driada")
sys.path.insert(0, str(DRIADA_ROOT / "src"))
sys.path.insert(0, str(DRIADA_ROOT / "tools"))

from load_synchronized_experiments import load_experiment_from_npz
from driada.information.info_base import MultiTimeSeries
from driada.intense.pipelines import compute_embedding_selectivity
from driada.intense.io import load_results

SYNC_DIR = DRIADA_ROOT / "DRIADA data" / "NOF" / "SynchronizedData26_v1"
INTENSE_DIR = DRIADA_ROOT / "DRIADA data" / "NOF" / "INTENSE_NOF_v3" / "results"
CACHE = Path(__file__).parent / "cache"

# Параметры
DS = 5            # временное прореживание
DIM = 10          # размерность вложения (коллективные переменные)
MIN_DIST = 0.8    # параметр UMAP (отчёт лаб. НИ 2023)
N_NEIGHBORS = 30
N_SHUF1 = 100
N_SHUF2 = 1000
PVAL_THR = 0.05
MULTICOMP = "fdr_bh"   # holm/0.01 зарезал крупные сессии (920 нейронов -> 0)
SEED = 42


def list_nof_sessions():
    return sorted(p.stem.replace("_aligned", "")
                  for p in SYNC_DIR.glob("NOF_*_aligned.npz"))


def behavioral_selective(session):
    """Множество нейронов, значимых хотя бы к одной поведенческой переменной
    (готовый прогон INTENSE_NOF_v3, stage2)."""
    rp = INTENSE_DIR / f"{session}_results.npz"
    if not rp.exists():
        return None
    res = load_results(str(rp))
    sig = res.significance
    out = set()
    for nid in sig:
        if any(sig[nid][f].get("stage2", False) for f in sig[nid]):
            out.add(int(nid))
    return out


def run_one(session, force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{session}.npz"
    if out.exists() and not force:
        print(f"  {session}: already cached")
        return True
    path = SYNC_DIR / f"{session}_aligned.npz"
    if not path.exists():
        print(f"  {session}: SKIP (npz not found)")
        return False

    t0 = time.time()
    exp = load_experiment_from_npz(path, verbose=False)
    n_cells, n_frames = exp.n_cells, exp.n_frames

    # --- UMAP-вложение (нормированный сигнал, прореженный) ---
    ca_ds = exp.calcium.scdata[:, ::DS]
    mts = MultiTimeSeries(ca_ds, discrete=False)
    np.random.seed(SEED)
    emb = mts.get_embedding(method="umap", dim=DIM,
                            min_dist=MIN_DIST, n_neighbors=N_NEIGHBORS)
    coords = emb.coords            # (DIM, n_ds)
    n_ds = coords.shape[1]
    expected = -(-n_frames // DS)
    if n_ds != expected:
        print(f"  {session}: WARN coords {n_ds} != expected {expected}")

    comp_var = np.var(coords, axis=1)

    exp.store_embedding(coords.T, method_name="umap", data_type="calcium",
                        metadata={"ds": DS, "n_components": DIM})

    # --- INTENSE: нейрон <-> коллективная переменная ---
    emb_res = compute_embedding_selectivity(
        exp, embedding_methods=["umap"], data_type="calcium",
        mode="two_stage", n_shuffles_stage1=N_SHUF1, n_shuffles_stage2=N_SHUF2,
        pval_thr=PVAL_THR, multicomp_correction=MULTICOMP,
        find_optimal_delays=True, n_jobs=-1, seed=SEED, verbose=False)
    r = emb_res["umap"]
    ncomp = r["n_components"]
    comp_sel = r["component_selectivity"]
    stats = r["stats"]

    # упорядочивание коллективных переменных по убыванию числа связанных нейронов
    # (у UMAP layout нормирован, дисперсия координат для упорядочивания не годится)
    n_sig_native = np.array([len(comp_sel.get(c, [])) for c in range(ncomp)])
    order = np.argsort(n_sig_native)[::-1]

    # на компоненту: число значимых, список относительной ВИ (rel_me_beh)
    n_sig = np.zeros(ncomp, dtype=int)
    rel_mi_by_comp = []          # список массивов (для распределений)
    for rank, comp in enumerate(order[:ncomp]):
        feat = f"umap_comp{comp}"
        nids = comp_sel.get(comp, [])
        n_sig[rank] = len(nids)
        vals = [stats[nid][feat]["rel_me_beh"] for nid in nids
                if feat in stats.get(nid, {})]
        rel_mi_by_comp.append(np.array(vals, dtype=float))

    emb_selective = set(int(n) for n in r["significant_neurons"].keys())

    # --- перекрытие с поведенческой селективностью ---
    beh_sel = behavioral_selective(session)
    if beh_sel is None:
        frac_emb_also_beh = np.nan
        n_overlap = -1
    else:
        overlap = emb_selective & beh_sel
        n_overlap = len(overlap)
        frac_emb_also_beh = (len(overlap) / len(emb_selective)
                             if emb_selective else np.nan)

    # ragged rel_mi -> object array
    rel_obj = np.empty(ncomp, dtype=object)
    for i, a in enumerate(rel_mi_by_comp):
        rel_obj[i] = a

    np.savez(out,
             session=session, n_cells=n_cells, n_components=ncomp,
             n_sig=n_sig, frac_sig=n_sig / max(n_cells, 1),
             comp_var=comp_var[order][:ncomp],
             rel_mi=rel_obj,
             emb_selective=np.array(sorted(emb_selective)),
             n_emb_sel=len(emb_selective),
             n_beh_sel=(-1 if beh_sel is None else len(beh_sel)),
             n_overlap=n_overlap,
             frac_emb_also_beh=frac_emb_also_beh)
    dt = time.time() - t0
    print(f"  {session}: emb-sel={len(emb_selective)} "
          f"beh-sel={'-' if beh_sel is None else len(beh_sel)} "
          f"overlap={frac_emb_also_beh:.2f} "
          f"frac_sig[0]={n_sig[0]/max(n_cells,1):.3f} ({dt:.0f}s)")
    return True


def run_all(force=False):
    sessions = list_nof_sessions()
    print(f"=== popcode: {len(sessions)} NOF sessions ===")
    print(f"DS={DS} DIM={DIM} min_dist={MIN_DIST} nn={N_NEIGHBORS} "
          f"shuf={N_SHUF1}/{N_SHUF2} pval={PVAL_THR} {MULTICOMP}")
    t = time.time()
    ok = fail = 0
    for i, s in enumerate(sessions, 1):
        print(f"\n[{i}/{len(sessions)}] {s}")
        try:
            ok += run_one(s, force=force)
        except Exception as e:
            import traceback
            print(f"  {s}: FAILED — {type(e).__name__}: {e}")
            traceback.print_exc()
            fail += 1
    print(f"\n=== done: {ok} ok, {fail} failed, {(time.time()-t)/60:.1f} min ===")


def aggregate():
    files = sorted(CACHE.glob("NOF_*.npz"))
    if not files:
        print("no caches")
        return
    frac_sig = []      # (n_sessions, ncomp)
    overlaps = []
    rel_all_by_comp = None
    ncomp_ref = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        nc = int(d["n_components"])
        if ncomp_ref is None:
            ncomp_ref = nc
            rel_all_by_comp = [[] for _ in range(nc)]
        if nc != ncomp_ref:
            continue
        frac_sig.append(d["frac_sig"])
        ov = float(d["frac_emb_also_beh"])
        if not np.isnan(ov):
            overlaps.append(ov)
        rel = d["rel_mi"]
        for c in range(nc):
            rel_all_by_comp[c].append(np.asarray(rel[c], dtype=float))
    frac_sig = np.array(frac_sig)          # (S, ncomp)
    mean_fs = frac_sig.mean(0) * 100
    # 95% CI по сессиям
    se = frac_sig.std(0, ddof=1) / np.sqrt(frac_sig.shape[0]) * 100
    ci = 1.96 * se
    overlaps = np.array(overlaps)

    print(f"\n=== AGGREGATE ({len(files)} sessions) ===")
    print("Доля значимых нейронов на коллективную переменную (%, mean±95%CI):")
    for c in range(ncomp_ref):
        print(f"  umap {c+1}: {mean_fs[c]:.2f} ± {ci[c]:.2f}")
    print(f"\nПерекрытие с поведенческой селективностью: "
          f"{overlaps.mean()*100:.1f}% ± {overlaps.std(ddof=1)*100:.1f}% "
          f"(n={len(overlaps)})")
    print(f"  => ~{overlaps.mean()*100:.0f}% нейронов попул. кода поведенчески селективны; "
          f"~{(1-overlaps.mean())*100:.0f}% — нет")

    rel_pooled = [np.concatenate(rel_all_by_comp[c]) if rel_all_by_comp[c] else np.array([])
                  for c in range(ncomp_ref)]
    rel_obj = np.empty(ncomp_ref, dtype=object)
    for c in range(ncomp_ref):
        rel_obj[c] = rel_pooled[c]
    np.savez(Path(__file__).parent / "popcode_summary.npz",
             mean_frac_sig=mean_fs, ci_frac_sig=ci,
             overlap_mean=overlaps.mean(), overlap_std=overlaps.std(ddof=1),
             n_sessions=len(files), rel_mi_pooled=rel_obj)
    print(f"\nsaved popcode_summary.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session")
    p.add_argument("--all", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.session:
        run_one(a.session, force=a.force)
    if a.all:
        run_all(force=a.force)
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
