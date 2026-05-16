"""
Шаг 3: Размерность представлений по слоям для фиксированных эпох.

Проверяем наличие «горба» (hunchback, Ansuini et al. 2019):
  - ВР (2-NN) по слоям: ожидается рост → пик → падение
  - ЭР (participation ratio) по слоям: для сравнения (линейная размерность)

Данные: mango_data/results/dimensionality.npz (предрассчитано в 02_dimensionality.py)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
from driada.intense.visual import make_beautiful

# ============================================================================
# Загрузка данных
# ============================================================================
OUT_DIR = Path(r'C:\Users\User\PycharmProjects\thesis\mango_data\results')

data = np.load(OUT_DIR / 'dimensionality.npz', allow_pickle=True)
ed = data['ed']       # (17, 26)
id_nn = data['id_nn'] # (17, 26)
LAYERS = [str(x) for x in data['layers']]
EPOCHS = [int(x) for x in data['epochs']]

# Короткие подписи слоёв
SHORT_LABELS = []
for l in LAYERS:
    if l == 'sn1':
        SHORT_LABELS.append('sn1')
    else:
        parts = l.split('.')
        block = parts[0].replace('layer', 'L')
        SHORT_LABELS.append(f'{block}.{parts[1]}.{parts[2][-1]}')


# ============================================================================
# Графики: размерность по слоям для нескольких эпох
# ============================================================================

SHOW_EPOCHS = [5, 20, 50, 100, 200, 400, 700, 900]


def plot_dim_across_layers():
    epoch_indices = [EPOCHS.index(e) for e in SHOW_EPOCHS]

    cmap = plt.get_cmap('coolwarm')
    clrs = cmap(np.linspace(0, 1, len(SHOW_EPOCHS)))

    x = np.arange(len(LAYERS))

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    for ax_idx, (dim_data, title, ylabel) in enumerate([
        (ed, 'Эффективная размерность (participation ratio) по слоям',
         'Эффективная размерность'),
        (id_nn, 'Внутренняя размерность (2-NN) по слоям',
         'Внутренняя размерность')
    ]):
        ax = make_beautiful(axes[ax_idx])
        for ci, (ei, ep) in enumerate(zip(epoch_indices, SHOW_EPOCHS)):
            vals = dim_data[:, ei]
            mask = ~np.isnan(vals)
            ax.plot(x[mask], vals[mask], 'o-', c=clrs[ci], lw=2.5, ms=6,
                    label=f'эпоха {ep}', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LABELS, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Слой (вход → выход)', fontsize=11)
        ax.legend(fontsize=9, ncol=4, loc='upper left')
        ax.set_title(title, fontsize=13)

        # Разделители между группами слоёв
        for sep in [0.5, 4.5, 8.5, 12.5]:
            ax.axvline(sep, c='grey', ls=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'dim_across_layers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "dim_across_layers.png"}')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f'Epochs available: {EPOCHS}')
    print(f'Layers: {len(LAYERS)}')
    print(f'Showing epochs: {SHOW_EPOCHS}')

    plot_dim_across_layers()

    print('\nDone!')
