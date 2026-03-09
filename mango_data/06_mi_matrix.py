"""
Шаг 6: MI-матрица между нейронами и её размерность.

Для каждого слоя × эпохи:
  - M_ij = I(t_i; t_j) — попарная MI между нейронами (exact discrete, с binning)
  - PR(M) — participation ratio MI-матрицы (нелинейная эффективная размерность)
  - Сравнение с ЭР (PR ковариации, линейная размерность)

Данные: mango_data/Activity/SJ-SNN-50/
"""

import sys
import gc
import numpy as np
from pathlib import Path
from sklearn.metrics import mutual_info_score

sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')

# ============================================================================
# Параметры
# ============================================================================
DATA_ROOT = Path(r'C:\Users\User\PycharmProjects\thesis\mango_data\Activity')
SNN_DIR = DATA_ROOT / 'SJ-SNN-50'
OUT_DIR = Path(r'C:\Users\User\PycharmProjects\thesis\mango_data\results')

EPOCHS = sorted([int(d.name.split()[-1]) for d in SNN_DIR.iterdir() if d.is_dir()])
LAYERS = [
    'sn1',
    'layer1.0.sn1', 'layer1.0.sn2', 'layer1.1.sn1', 'layer1.1.sn2',
    'layer2.0.sn1', 'layer2.0.sn2', 'layer2.1.sn1', 'layer2.1.sn2',
    'layer3.0.sn1', 'layer3.0.sn2', 'layer3.1.sn1', 'layer3.1.sn2',
    'layer4.0.sn1', 'layer4.0.sn2', 'layer4.1.sn1', 'layer4.1.sn2',
]

LN2 = np.log(2)
N_BINS = 10  # бинов для дискретизации firing rates


def load_activity(epoch, layer, split='test'):
    epoch_dir = SNN_DIR / f'iter {epoch}' / split
    fname = f'SJ-SNN act iter {epoch} {split} layer {layer}.npy.npz'
    if layer == 'sn1':
        fname = f'SJ-SNN act iter {epoch} {split} layer sn1.npy.npz'
    path = epoch_dir / fname
    if not path.exists():
        return None
    arr = np.load(path)['arr_0']  # (T, n_images, N)
    return arr.sum(axis=0)  # (n_images, N)


def bin_activity(act, n_bins=N_BINS):
    """Бинирование firing rates для MI-оценки.

    Используем квантили, чтобы бины были равнонаполненными.
    """
    binned = np.zeros_like(act, dtype=np.int32)
    for i in range(act.shape[1]):
        col = act[:, i]
        # Если все значения одинаковые — один бин
        if col.max() == col.min():
            binned[:, i] = 0
        else:
            # Квантильный бининг
            percentiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(col, percentiles)
            edges = np.unique(edges)  # убрать дубликаты
            binned[:, i] = np.digitize(col, edges[1:-1])
    return binned


def compute_mi_matrix_gauss(act):
    """MI-матрица через гауссову аппроксимацию: I = -0.5·log(1-r²).

    Быстрее discrete MI и не требует бинирования.
    Используем Spearman r (ранговый) для robustness.
    """
    import warnings
    from scipy.stats import spearmanr
    N = act.shape[1]
    if N < 2:
        return np.zeros((1, 1))

    # Убрать constant columns (мёртвые нейроны)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(act)

    # spearmanr может вернуть скаляр для N=2
    rho = np.atleast_2d(rho)

    # NaN → 0 (constant columns → no correlation)
    rho = np.nan_to_num(rho, nan=0.0)

    # MI = -0.5 * log2(1 - r^2)
    rho2 = np.clip(rho ** 2, 0, 1 - 1e-15)
    M = -0.5 * np.log2(1 - rho2)
    np.fill_diagonal(M, 0)  # диагональ = 0 для PR
    return M


def participation_ratio(matrix):
    """PR собственных значений матрицы."""
    eigvals = np.linalg.eigvalsh(matrix)
    eigvals = eigvals[eigvals > 0]  # только положительные
    if len(eigvals) == 0:
        return np.nan
    return np.sum(eigvals) ** 2 / np.sum(eigvals ** 2)


# ============================================================================
# Основной расчёт
# ============================================================================

def compute_all():
    n_layers = len(LAYERS)
    n_epochs = len(EPOCHS)

    pr_mi = np.full((n_layers, n_epochs), np.nan)

    for j, epoch in enumerate(EPOCHS):
        print(f'  Epoch {epoch}...', end=' ', flush=True)
        for i, layer in enumerate(LAYERS):
            act = load_activity(epoch, layer)
            if act is None:
                continue

            M = compute_mi_matrix_gauss(act)
            pr_mi[i, j] = participation_ratio(M)

            del act, M
            gc.collect()
        print('done')

    return pr_mi


# ============================================================================
# Визуализация
# ============================================================================

def plot_comparison(pr_mi):
    """PR(MI matrix) vs ЭР по эпохам."""
    import matplotlib.pyplot as plt
    from driada.intense.visual import make_beautiful

    dim_data = np.load(OUT_DIR / 'dimensionality.npz', allow_pickle=True)
    ed = dim_data['ed']  # ЭР = PR ковариации

    groups = [
        ('sn1 + layer1', [0, 1, 2, 3, 4]),
        ('layer2', [5, 6, 7, 8]),
        ('layer3', [9, 10, 11, 12]),
        ('layer4', [13, 14, 15, 16]),
    ]

    SHORT = []
    for l in LAYERS:
        if l == 'sn1':
            SHORT.append('sn1')
        else:
            parts = l.split('.')
            block = parts[0].replace('layer', 'L')
            SHORT.append(f'{block}.{parts[1]}.{parts[2][-1]}')

    mask_ep = np.array(EPOCHS) > 0
    xdata = np.log10(np.array(EPOCHS)[mask_ep] + 1)

    # Два ряда: верхний — PR(MI), нижний — ЭР
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    cmap = plt.get_cmap('plasma')

    for col, (group_name, indices) in enumerate(groups):
        clrs = cmap(np.linspace(0.1, 0.9, len(indices)))

        # PR(MI matrix)
        ax = make_beautiful(axes[0, col])
        for ci, idx in enumerate(indices):
            vals = pr_mi[idx, mask_ep]
            valid = ~np.isnan(vals)
            if valid.any():
                ax.plot(xdata[valid], vals[valid], 'o-', c=clrs[ci],
                        lw=2.5, ms=5, label=SHORT[idx])
        ax.set_title(f'{group_name}: PR(MI matrix)', fontsize=11)
        ax.set_ylabel('PR(MI)')
        ax.legend(fontsize=8)

        # ЭР
        ax2 = make_beautiful(axes[1, col])
        for ci, idx in enumerate(indices):
            vals = ed[idx, mask_ep]
            valid = ~np.isnan(vals)
            if valid.any():
                ax2.plot(xdata[valid], vals[valid], 'o-', c=clrs[ci],
                         lw=2.5, ms=5, label=SHORT[idx])
        ax2.set_title(f'{group_name}: ЭР (PR covariance)', fontsize=11)
        ax2.set_xlabel(r'$\log_{10}(\mathrm{epoch}+1)$')
        ax2.set_ylabel('ЭР')
        ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'mi_matrix_vs_er.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "mi_matrix_vs_er.png"}')

    # Scatter: PR(MI) vs ЭР
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax = make_beautiful(ax)

    layer_groups = {
        'layer1': ([1, 2, 3, 4], 'blue'),
        'layer2': ([5, 6, 7, 8], 'green'),
        'layer3': ([9, 10, 11, 12], 'orange'),
        'layer4': ([13, 14, 15, 16], 'red'),
    }

    for gname, (indices, color) in layer_groups.items():
        for idx in indices:
            x = ed[idx, mask_ep]
            y = pr_mi[idx, mask_ep]
            valid = ~np.isnan(x) & ~np.isnan(y)
            if valid.any():
                ax.scatter(x[valid], y[valid], c=color, s=20, alpha=0.5,
                           label=gname if idx == indices[0] else None)

    # Диагональ
    lims = [0, max(np.nanmax(ed), np.nanmax(pr_mi)) * 1.1]
    ax.plot(lims, lims, '--', c='grey', alpha=0.5, label='y=x')

    ax.set_xlabel('ЭР (PR covariance)', fontsize=12)
    ax.set_ylabel('PR(MI matrix)', fontsize=12)
    ax.set_title('Linear vs nonlinear effective dimensionality', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'pr_mi_vs_er_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "pr_mi_vs_er_scatter.png"}')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f'Epochs: {EPOCHS}')
    print(f'Layers: {len(LAYERS)}')

    npz_path = OUT_DIR / 'mi_matrix.npz'
    if npz_path.exists():
        print('Loading precomputed MI matrix results...')
        data = np.load(npz_path)
        pr_mi = data['pr_mi']
    else:
        print('\n=== Computing MI matrices ===')
        pr_mi = compute_all()
        np.savez(npz_path, pr_mi=pr_mi, layers=LAYERS, epochs=EPOCHS)
        print(f'Saved: {npz_path}')

    print('\n=== Plotting ===')
    plot_comparison(pr_mi)

    print('\nDone!')
