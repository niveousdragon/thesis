"""
Шаг 4: Сопоставление V-shape ВР с метриками кластеризации.

Гипотеза: слои с V-shape ВР — те же, где CH score продолжает расти
(т.е. разделение классов продолжается), а в остальных — уходит.

Данные:
  - results/dimensionality.npz (ЭР, ВР)
  - results/clustering_metrics_umap.npz (силуэт, CH, DB)
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

dim_data = np.load(OUT_DIR / 'dimensionality.npz', allow_pickle=True)
ed = dim_data['ed']       # (17, 26)
id_nn = dim_data['id_nn'] # (17, 26)
LAYERS = [str(x) for x in dim_data['layers']]
EPOCHS = [int(x) for x in dim_data['epochs']]

clust_data = np.load(OUT_DIR / 'clustering_metrics_umap.npz', allow_pickle=True)
ch = clust_data['calinski_harabasz']   # (17, 26)
sil = clust_data['silhouette']         # (17, 26)
db = clust_data['davies_bouldin']      # (17, 26)

# Короткие подписи слоёв
SHORT = []
for l in LAYERS:
    if l == 'sn1':
        SHORT.append('sn1')
    else:
        parts = l.split('.')
        block = parts[0].replace('layer', 'L')
        SHORT.append(f'{block}.{parts[1]}.{parts[2][-1]}')

# Без эпохи 0
mask_ep = np.array(EPOCHS) > 0
epochs_no0 = np.array(EPOCHS)[mask_ep]
xdata = np.log10(epochs_no0 + 1)


# ============================================================================
# Совмещённые графики: ВР + CH для каждого слоя
# ============================================================================

def plot_vshape_vs_ch():
    """Два ряда: ВР и CH, каждый слой — отдельная кривая, группировка по блокам."""

    # Группы слоёв
    groups = [
        ('sn1 + layer1', [0, 1, 2, 3, 4]),
        ('layer2', [5, 6, 7, 8]),
        ('layer3', [9, 10, 11, 12]),
        ('layer4', [13, 14, 15, 16]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    cmap = plt.get_cmap('plasma')

    for col, (group_name, indices) in enumerate(groups):
        clrs = cmap(np.linspace(0.1, 0.9, len(indices)))

        # Верхний ряд: ВР
        ax_id = make_beautiful(axes[0, col])
        for ci, idx in enumerate(indices):
            vals = id_nn[idx, mask_ep]
            valid = ~np.isnan(vals)
            if valid.any():
                ax_id.plot(xdata[valid], vals[valid], 'o-', c=clrs[ci],
                           lw=2.5, ms=5, label=SHORT[idx])
                # Отметить минимум
                min_i = np.nanargmin(vals)
                if not np.isnan(vals[min_i]):
                    ax_id.plot(xdata[min_i], vals[min_i], 's', c=clrs[ci],
                               ms=12, markeredgecolor='k', markeredgewidth=2, zorder=10)
        ax_id.set_title(f'{group_name}: Intrinsic dim (2-NN)', fontsize=11)
        ax_id.set_ylabel('ВР')
        ax_id.legend(fontsize=8)

        # Нижний ряд: CH
        ax_ch = make_beautiful(axes[1, col])
        for ci, idx in enumerate(indices):
            vals = ch[idx, mask_ep]
            valid = ~np.isnan(vals)
            if valid.any():
                ax_ch.plot(xdata[valid], vals[valid], 'o-', c=clrs[ci],
                           lw=2.5, ms=5, label=SHORT[idx])
        ax_ch.set_title(f'{group_name}: CH score', fontsize=11)
        ax_ch.set_xlabel(r'$\log_{10}(\mathrm{epoch}+1)$')
        ax_ch.set_ylabel('CH')
        ax_ch.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'vshape_vs_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "vshape_vs_clustering.png"}')


# ============================================================================
# Количественная сводка: где V-shape, где рост CH
# ============================================================================

def print_summary():
    """Для каждого слоя: есть ли минимум ВР? Растёт ли CH на поздних эпохах?"""
    print(f'\n{"Layer":16s} {"ID min ep":>10s} {"ID min val":>10s} {"ID final":>10s}'
          f' {"CH trend":>10s} {"CH final":>10s}')
    print('-' * 70)

    for i, layer in enumerate(LAYERS):
        vals_id = id_nn[i, mask_ep]
        vals_ch = ch[i, mask_ep]

        # Минимум ВР (исключая NaN)
        valid_id = ~np.isnan(vals_id)
        if valid_id.sum() < 3:
            print(f'{SHORT[i]:16s} {"N/A":>10s}')
            continue

        min_idx = np.nanargmin(vals_id)
        min_ep = epochs_no0[min_idx]
        min_val = vals_id[min_idx]
        final_val = vals_id[valid_id][-1]

        # V-shape: минимум не на краях
        is_vshape = 1 < min_idx < valid_id.sum() - 2

        # CH тренд: сравнить среднее последних 5 с серединой
        valid_ch = ~np.isnan(vals_ch)
        if valid_ch.sum() > 10:
            mid = valid_ch.sum() // 2
            ch_mid = np.nanmean(vals_ch[valid_ch][:mid])
            ch_late = np.nanmean(vals_ch[valid_ch][-5:])
            ch_trend = 'grows' if ch_late > ch_mid * 1.1 else ('flat' if ch_late > ch_mid * 0.9 else 'drops')
            ch_final = vals_ch[valid_ch][-1]
        else:
            ch_trend = 'N/A'
            ch_final = np.nan

        vmark = '  <<<' if is_vshape else ''
        print(f'{SHORT[i]:16s} {min_ep:10d} {min_val:10.1f} {final_val:10.1f}'
              f' {ch_trend:>10s} {ch_final:10.0f}{vmark}')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print_summary()
    plot_vshape_vs_ch()
    print('\nDone!')
