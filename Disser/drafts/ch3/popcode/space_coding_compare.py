#!/usr/bin/env python
"""Кто лучше кодирует пространство: AE (lambda=0) или UMAP?

Честное сравнение на ОДИНАКОВОМ входе: matched-клетки, те же valid-кадры,
та же размерность (16). Из каждого вложения декодируем положение (x,y)
KNN-регрессором с 5-fold CV; ошибка в см (медиана). Меньше = лучше.

Вложения:
  AE   — готовый lambda=0 латент (data_matched_ds5), 16D
  UMAP — пересчитываем на тех же matched-клетках, dim=16, min_dist=0.8
  PCA  — линейный базлайн, 16 компонент
  *_shuf — на per-neuron roll-перемешанном кальции (уровень случайности)

Usage: python space_coding_compare.py           # все 64 сессии
       python space_coding_compare.py --session NOF_H01_1D
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, argparse, warnings, time
from pathlib import Path
import numpy as np

DRIADA = Path(r"C:\Users\User\PycharmProjects\driada")
sys.path.insert(0, str(DRIADA / "src"))
from driada.information.info_base import MultiTimeSeries
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

IB = Path(r"C:\Users\User\PycharmProjects\infobottleneck")
AE_DIR = IB / "results" / "nof" / "data_matched_ds5"
NOF_DATA = DRIADA / "science" / "NOF data"
DIM = 16
SEED = 42


def list_sessions():
    return ["NOF_" + p.stem.replace("pilot_matched_", "")
            for p in sorted(AE_DIR.glob("pilot_matched_*.npz"))]


def umap_embed(ca, dim=DIM):
    """ca: (n_neurons, n_valid) -> coords (n_valid, dim)."""
    mts = MultiTimeSeries(ca, discrete=False)
    np.random.seed(SEED)
    emb = mts.get_embedding(method="umap", dim=dim, min_dist=0.8, n_neighbors=30)
    return emb.coords.T


def decode_err(emb, pos, k=10):
    """Медианная ошибка декодирования позиции (см), 5-fold CV, KNN."""
    Xs = StandardScaler().fit_transform(emb)
    pred = cross_val_predict(KNeighborsRegressor(k), Xs, pos,
                             cv=KFold(5, shuffle=False))
    return float(np.median(np.sqrt(((pred - pos) ** 2).sum(1))))


def roll_shuffle(ca, seed):
    rng = np.random.default_rng(seed)
    return np.array([np.roll(c, int(rng.integers(1, ca.shape[1] - 1))) for c in ca])


def run_one(session):
    _, mouse, day = session.split("_")
    z = np.load(AE_DIR / f"pilot_matched_{mouse}_{day}.npz")
    Z = np.vstack([z["Z_tr"], z["Z_te"]]).astype(np.float64)        # (n_valid,16)
    Z_shuf = np.vstack([z["Z_tr_shuf"][0], z["Z_te_shuf"][0]]).astype(np.float64)
    pos = np.vstack([z["Y_tr"], z["Y_te"]]).astype(np.float64)      # (n_valid,2) x,y
    valid_idx = z["valid_idx"].astype(int)
    matched_idx = z["matched_idx"].astype(int)

    sess = np.load(NOF_DATA / f"NOF_{mouse}_{day} syn data.npz")
    ca = sess["calcium"][matched_idx, :][:, valid_idx]             # (n_matched,n_valid)

    res = {}
    res["AE"] = decode_err(Z, pos)
    res["UMAP"] = decode_err(umap_embed(ca), pos)
    res["PCA"] = decode_err(PCA(DIM, random_state=SEED).fit_transform(ca.T), pos)
    res["AE_shuf"] = decode_err(Z_shuf, pos)
    res["UMAP_shuf"] = decode_err(umap_embed(roll_shuffle(ca, SEED)), pos)
    # chance: предсказание глобальным средним положением
    res["chance"] = float(np.median(np.sqrt(((pos - pos.mean(0)) ** 2).sum(1))))
    res["n_cells"] = ca.shape[0]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session")
    a = p.parse_args()
    sessions = [a.session] if a.session else list_sessions()
    rows = {}
    t = time.time()
    for i, s in enumerate(sessions, 1):
        try:
            r = run_one(s)
            rows[s] = r
            print(f"[{i}/{len(sessions)}] {s}: AE={r['AE']:.3f} UMAP={r['UMAP']:.3f} "
                  f"PCA={r['PCA']:.3f} | chance={r['chance']:.3f} "
                  f"AEsh={r['AE_shuf']:.3f} UMsh={r['UMAP_shuf']:.3f} (n={r['n_cells']})")
        except Exception as e:
            import traceback; print(f"  {s}: FAIL {type(e).__name__}: {e}")
            traceback.print_exc()
    if len(rows) < 2:
        return
    keys = ["AE", "UMAP", "PCA", "AE_shuf", "UMAP_shuf", "chance"]
    arr = {k: np.array([rows[s][k] for s in rows]) for k in keys}
    print(f"\n=== Ошибка декодирования положения, норм. ед. [0,1] "
          f"(медиана по {len(rows)} сессиям; арена 44 см) ===")
    for k in keys:
        cm = np.median(arr[k]) * 44
        print(f"  {k:10s}: {np.median(arr[k]):.3f}  (~{cm:.1f} см)  "
              f"mean {arr[k].mean():.3f} ± {arr[k].std(ddof=1):.3f}")
    # парный тест AE vs UMAP
    stat, pv = wilcoxon(arr["AE"], arr["UMAP"])
    better = "AE" if np.median(arr["AE"]) < np.median(arr["UMAP"]) else "UMAP"
    n_ae = int((arr["AE"] < arr["UMAP"]).sum())
    print(f"\nAE vs UMAP: лучше (меньше ошибка) — {better}; "
          f"AE точнее в {n_ae}/{len(rows)} сессиях; Wilcoxon p={pv:.2e}")
    np.savez(Path(__file__).parent / "space_coding_compare.npz",
             sessions=np.array(list(rows.keys())),
             **{k: arr[k] for k in keys})
    print(f"({(time.time()-t)/60:.1f} мин) saved space_coding_compare.npz")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
