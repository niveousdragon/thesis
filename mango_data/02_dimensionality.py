"""
Шаг 2: Размерность представлений SNN по слоям и эпохам.

- ЭР (эффективная размерность, participation ratio) — DRIADA eff_dim
- ВР (внутренняя размерность, 2-NN) — DRIADA nn_dimension

Данные: mango_data/Activity/SJ-SNN-50/
"""

import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
from driada.dimensionality import eff_dim, nn_dimension
from driada.intense.visual import make_beautiful

# ============================================================================
# Параметры (те же, что в 01_reproduce_report.py)
# ============================================================================
DATA_ROOT = Path(r'C:\Users\User\PycharmProjects\thesis\mango_data\Activity')
SNN_DIR = DATA_ROOT / 'SJ-SNN-50'
OUT_DIR = Path(r'C:\Users\User\PycharmProjects\thesis\mango_data\results')
OUT_DIR.mkdir(exist_ok=True)

EPOCHS = sorted([int(d.name.split()[-1]) for d in SNN_DIR.iterdir() if d.is_dir()])
LAYERS = [
    'sn1',
    'layer1.0.sn1', 'layer1.0.sn2', 'layer1.1.sn1', 'layer1.1.sn2',
    'layer2.0.sn1', 'layer2.0.sn2', 'layer2.1.sn1', 'layer2.1.sn2',
    'layer3.0.sn1', 'layer3.0.sn2', 'layer3.1.sn1', 'layer3.1.sn2',
    'layer4.0.sn1', 'layer4.0.sn2', 'layer4.1.sn1', 'layer4.1.sn2',
]


def load_activity(epoch, layer, split='test'):
    epoch_dir = SNN_DIR / f'iter {epoch}' / split
    fname = f'SJ-SNN act iter {epoch} {split} layer {layer}.npy.npz'
    if layer == 'sn1':
        fname = f'SJ-SNN act iter {epoch} {split} layer sn1.npy.npz'
    path = epoch_dir / fname
    if not path.exists():
        return None
    arr = np.load(path)['arr_0']  # (T, n_images, N)
    return arr.sum(axis=0)  # (n_images, N) — firing rates


# ============================================================================
# Вычисление размерностей
# ============================================================================

def compute_dimensionalities():
    """ЭР и ВР для всех слоёв × эпох."""
    ed = np.full((len(LAYERS), len(EPOCHS)), np.nan)
    id_nn = np.full((len(LAYERS), len(EPOCHS)), np.nan)

    for j, epoch in enumerate(EPOCHS):
        print(f'  Epoch {epoch}...', end=' ', flush=True)
        for i, layer in enumerate(LAYERS):
            act = load_activity(epoch, layer)
            if act is None:
                continue

            # act shape: (n_samples=10000, n_features=N)
            # eff_dim и nn_dimension ожидают (n_samples, n_features)

            # ЭР: participation ratio
            try:
                ed[i, j] = eff_dim(act, enable_correction=True, q=2)
            except Exception as e:
                print(f'ED error {layer} ep{epoch}: {e}')

            # ВР: 2-NN (подвыборка для скорости)
            try:
                n_sub = min(5000, act.shape[0])
                idx = np.random.RandomState(42).choice(act.shape[0], n_sub, replace=False)
                id_nn[i, j] = nn_dimension(act[idx], k=2)
            except Exception as e:
                print(f'ID error {layer} ep{epoch}: {e}')

            del act
            gc.collect()
        print('done')

    return ed, id_nn


# ============================================================================
# Визуализация
# ============================================================================

def plot_heatmaps(ed, id_nn):
    """Тепловые карты: слой × эпоха для ЭР и ВР."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, data, title in zip(axes,
                                [ed, id_nn],
                                ['Effective dim (participation ratio)',
                                 'Intrinsic dim (2-NN)']):
        im = ax.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels(LAYERS, fontsize=8)
        ax.set_xticks(range(len(EPOCHS)))
        ax.set_xticklabels(EPOCHS, rotation=90, fontsize=7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Layer')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'dimensionality_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "dimensionality_heatmaps.png"}')


def plot_dim_curves(ed, id_nn):
    """Линейные графики размерности по эпохам (стиль colab)."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.set_tight_layout(True)

    cmap = plt.get_cmap('plasma')
    clrs = cmap(np.linspace(0, 1.0, len(LAYERS)))
    xdata = np.log10(np.array(EPOCHS) + 1)

    for ax, data, title in zip(axes,
                                [ed, id_nn],
                                ['Effective dim (PR)', 'Intrinsic dim (2-NN)']):
        ax = make_beautiful(ax)
        for i in range(len(LAYERS)):
            ax.plot(xdata, data[i, :], c=clrs[i], lw=3,
                    label=LAYERS[i] if i % 4 == 0 else None)
        ax.set_xlabel(r'$\log_{10}(\mathrm{epoch}+1)$')
        ax.set_ylabel(title)
        ax.legend()

    plt.savefig(OUT_DIR / 'dimensionality_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "dimensionality_curves.png"}')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f'Epochs: {EPOCHS}')
    print(f'Layers: {len(LAYERS)}')

    npz_path = OUT_DIR / 'dimensionality.npz'
    if npz_path.exists():
        print('Loading precomputed dimensionalities...')
        data = np.load(npz_path)
        ed, id_nn = data['ed'], data['id_nn']
    else:
        print('\n=== Computing dimensionalities ===')
        ed, id_nn = compute_dimensionalities()
        np.savez(npz_path, ed=ed, id_nn=id_nn,
                 layers=LAYERS, epochs=EPOCHS)
        print(f'Saved: {npz_path}')

    print('\n=== Plotting ===')
    plot_heatmaps(ed, id_nn)
    plot_dim_curves(ed, id_nn)

    print('\nDone!')
