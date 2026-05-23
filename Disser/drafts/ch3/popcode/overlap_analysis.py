#!/usr/bin/env python
"""Перекрытие популяционного кода и поведенческой селективности под разными
определениями. Читает cache/{session}.npz (emb_selective) и готовый
INTENSE_NOF_v3. Не перезапускает INTENSE.

Определения:
  A) полный поведенческий набор v3 (incl. place)
  B) только непространственные фичи (без place/walls/center/corners)
  C) обогащение над базой: наблюдаемое перекрытие / ожидаемое при случайности
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, r"C:\Users\User\PycharmProjects\driada\src")
sys.path.insert(0, r"C:\Users\User\PycharmProjects\driada\tools")
from driada.intense.io import load_results

HERE = Path(__file__).parent
CACHE = HERE / "cache"
INTENSE_DIR = Path(r"C:\Users\User\PycharmProjects\driada\DRIADA data\NOF\INTENSE_NOF_v3\results")

SPATIAL = {"place", "walls", "center", "corners"}


def beh_selective_sets(session):
    """(все селективные, непространственно-селективные)."""
    rp = INTENSE_DIR / f"{session}_results.npz"
    if not rp.exists():
        return None, None
    sig = load_results(str(rp)).significance
    allset, nonspat = set(), set()
    for nid in sig:
        feats_sig = [f for f in sig[nid] if sig[nid][f].get("stage2", False)]
        if feats_sig:
            allset.add(int(nid))
        if any(f not in SPATIAL for f in feats_sig):
            nonspat.add(int(nid))
    return allset, nonspat


def main():
    files = sorted(CACHE.glob("NOF_*.npz"))
    A, B, C = [], [], []   # overlap full, overlap nonspatial, enrichment
    base_rates = []
    n_used = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        session = str(d["session"])
        emb = set(int(x) for x in d["emb_selective"])
        n_cells = int(d["n_cells"])
        if not emb:
            continue
        ball, bnon = beh_selective_sets(session)
        if ball is None:
            continue
        n_used += 1
        ov_all = len(emb & ball) / len(emb)
        ov_non = len(emb & bnon) / len(emb)
        base = len(ball) / n_cells
        A.append(ov_all)
        B.append(ov_non)
        base_rates.append(base)
        C.append(ov_all / base if base > 0 else np.nan)
    A, B, C = np.array(A), np.array(B), np.array(C)
    base_rates = np.array(base_rates)
    C = C[np.isfinite(C)]

    print(f"=== Перекрытие популяционного кода и поведения ({n_used} сессий) ===")
    print(f"Базовая доля поведенчески селективных (любая фича): "
          f"{base_rates.mean()*100:.0f}% ± {base_rates.std(ddof=1)*100:.0f}%")
    print()
    print(f"A) Полный набор v3 (incl. place):")
    print(f"     {A.mean()*100:.0f}% ± {A.std(ddof=1)*100:.0f}% нейронов попул. кода поведенчески селективны")
    print(f"B) Только непространственные фичи:")
    print(f"     {B.mean()*100:.0f}% ± {B.std(ddof=1)*100:.0f}%")
    print(f"C) Обогащение над базой (раз): {C.mean():.2f} ± {C.std(ddof=1):.2f}")
    print(f"     (>1 => нейроны попул. кода селективны чаще случайного)")


if __name__ == "__main__":
    main()
