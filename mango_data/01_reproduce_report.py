"""
Воспроизведение результатов отчёта лаб НИ 2024 (раздел 6):
  - PCA/UMAP-вложения послойной активности SNN
  - Метрики кластеризации (силуэт, Calinski-Harabasz, Davies-Bouldin)

Данные: mango_data/Activity/SJ-SNN-50/
"""

import sys
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pathlib import Path

sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
from driada import MVData
from driada.intense.visual import make_beautiful

# ============================================================================
# Параметры
# ============================================================================
DATA_ROOT = Path(r'C:\Users\User\PycharmProjects\thesis\mango_data\Activity')
SNN_DIR = DATA_ROOT / 'SJ-SNN-50'
OUT_DIR = Path(r'C:\Users\User\PycharmProjects\thesis\mango_data\results')
OUT_DIR.mkdir(exist_ok=True)

# Эпохи и слои
EPOCHS = sorted([int(d.name.split()[-1]) for d in SNN_DIR.iterdir() if d.is_dir()])
LAYERS = [
    'sn1',
    'layer1.0.sn1', 'layer1.0.sn2', 'layer1.1.sn1', 'layer1.1.sn2',
    'layer2.0.sn1', 'layer2.0.sn2', 'layer2.1.sn1', 'layer2.1.sn2',
    'layer3.0.sn1', 'layer3.0.sn2', 'layer3.1.sn1', 'layer3.1.sn2',
    'layer4.0.sn1', 'layer4.0.sn2', 'layer4.1.sn1', 'layer4.1.sn2',
]

LABEL_NAMES = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck',
}

CLASS_COLORS = ['r', 'g', 'b', 'c', 'violet', 'y', 'k', 'gold', 'purple', 'brown']

# ============================================================================
# Загрузка данных
# ============================================================================

def load_labels():
    f = np.load(DATA_ROOT / 'CIFAR test labels.npz')
    return f['arr_0']

def load_activity(epoch, layer, split='test'):
    """Загрузить активность одного слоя на одной эпохе.

    Возвращает (n_images, n_neurons) — суммарные спайки по timesteps.
    """
    epoch_dir = SNN_DIR / f'iter {epoch}' / split
    fname = f'SJ-SNN act iter {epoch} {split} layer {layer}.npy.npz'
    if layer == 'sn1':
        fname = f'SJ-SNN act iter {epoch} {split} layer sn1.npy.npz'
    path = epoch_dir / fname
    if not path.exists():
        return None
    arr = np.load(path)['arr_0']  # (T, n_images, N)
    # Суммируем по timesteps → firing rate
    return arr.sum(axis=0)  # (n_images, N)


# ============================================================================
# Визуализация эмбеддингов (стиль colab)
# ============================================================================

def plot_embedding_scatter(ax, coords, labels, title=None):
    """3D scatter с per-class цветами и легендой (стиль colab)."""
    ax = make_beautiful(ax)
    for lb in range(10):
        lb_inds = np.where(labels == lb)[0]
        ax.scatter(coords[0][lb_inds], coords[1][lb_inds], coords[2][lb_inds],
                   s=10, c=CLASS_COLORS[lb], label=LABEL_NAMES[lb], alpha=0.5)
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def plot_embedding_grid(epochs_to_show, layers_to_show, labels, method='pca',
                        n_components=3, out_name=None):
    """Сетка эмбеддингов: строки = слои, столбцы = эпохи."""
    n_rows = len(layers_to_show)
    n_cols = len(epochs_to_show)
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows), dpi=150)

    for i, layer in enumerate(layers_to_show):
        for j, epoch in enumerate(epochs_to_show):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection='3d')
            act = load_activity(epoch, layer)
            if act is None:
                ax.set_title(f'NO DATA\n{layer}, ep {epoch}')
                continue

            mvd = MVData(act.T, allow_zero_columns=True, rescale_rows=True)
            emb = mvd.get_embedding(method=method, n_components=n_components)
            coords = emb.coords  # (3, n_samples)

            plot_embedding_scatter(ax, coords, labels,
                                   title=f'{method.upper()} {layer}\nepoch {epoch}')

            del act, mvd, emb, coords
            gc.collect()

    # Общая легенда
    handles, leg_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc='upper center', ncol=5,
               fontsize=10, markerscale=3, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = out_name or f'{method}_grid.png'
    plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / fname}')


# ============================================================================
# Метрики кластеризации по слоям × эпохам
# ============================================================================

def compute_clustering_metrics(labels, method='umap', n_comp=10):
    """Считаем силуэт, CH, DB для всех слоёв и эпох.

    method: 'umap' (как в оригинальном colab) или 'pca'
    """
    results = {
        'silhouette': np.full((len(LAYERS), len(EPOCHS)), np.nan),
        'calinski_harabasz': np.full((len(LAYERS), len(EPOCHS)), np.nan),
        'davies_bouldin': np.full((len(LAYERS), len(EPOCHS)), np.nan),
    }

    for j, epoch in enumerate(EPOCHS):
        print(f'  Epoch {epoch}...', end=' ', flush=True)
        for i, layer in enumerate(LAYERS):
            act = load_activity(epoch, layer)
            if act is None:
                continue

            mvd = MVData(act.T, allow_zero_columns=True, rescale_rows=True)
            nc = min(n_comp, act.shape[1])
            emb = mvd.get_embedding(method=method, n_components=nc)
            X = emb.coords.T  # (n_samples, n_comp)

            # Подвыборка для скорости (силуэт тяжёлый)
            n_sub = min(5000, X.shape[0])
            idx = np.random.RandomState(42).choice(X.shape[0], n_sub, replace=False)
            X_sub = X[idx]
            labels_sub = labels[idx]

            try:
                results['silhouette'][i, j] = silhouette_score(X_sub, labels_sub)
                results['calinski_harabasz'][i, j] = calinski_harabasz_score(X_sub, labels_sub)
                results['davies_bouldin'][i, j] = davies_bouldin_score(X_sub, labels_sub)
            except Exception as e:
                print(f'Error {layer} ep{epoch}: {e}')
            del act, mvd, emb, X, X_sub
            gc.collect()
        print('done')

    return results


def plot_clustering_metrics(results, out_name='clustering_metrics_umap.png'):
    """Линейные графики метрик — все слои, plasma colormap (стиль colab)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.set_tight_layout(True)
    metric_names = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    titles = ['Silhouette coefficient', 'CH score', 'DB score']

    cmap = plt.get_cmap('plasma')
    clrs = cmap(np.linspace(0, 1.0, len(LAYERS)))

    for ax, metric, title in zip(axes, metric_names, titles):
        ax = make_beautiful(ax)
        for i in range(len(LAYERS)):
            vals = results[metric][i, :]
            ax.plot(np.log10(np.array(EPOCHS) + 1), vals,
                    c=clrs[i], lw=3,
                    label=LAYERS[i] if i % 4 == 0 else None)
        ax.set_xlabel(r'$\log_{10}(\mathrm{epoch}+1)$')
        ax.set_ylabel(title)
        ax.legend()

    plt.savefig(OUT_DIR / out_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / out_name}')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f'Epochs available: {EPOCHS}')
    print(f'Layers: {len(LAYERS)}')

    labels = load_labels()
    print(f'Labels: {labels.shape}, classes: {np.unique(labels)}')

    show_epochs = [0, 100, 500, 900]
    show_layers = ['sn1', 'layer2.1.sn2', 'layer4.1.sn2']

    # 1. PCA grid
    print('\n=== PCA вложения ===')
    plot_embedding_grid(show_epochs, show_layers, labels,
                        method='pca', out_name='pca_grid.png')

    # 2. UMAP grid
    print('\n=== UMAP вложения ===')
    plot_embedding_grid(show_epochs, show_layers, labels,
                        method='umap', out_name='umap_grid.png')

    # 3. Clustering metrics (UMAP-10)
    # Загрузим из файла если уже посчитаны
    metrics_path = OUT_DIR / 'clustering_metrics_umap.npz'
    if metrics_path.exists():
        print('\nМетрики уже посчитаны, перерисовываем')
        data = np.load(metrics_path)
        results = {k: data[k] for k in ['silhouette', 'calinski_harabasz', 'davies_bouldin']}
    else:
        print('\n=== Метрики кластеризации (UMAP-10) ===')
        results = compute_clustering_metrics(labels, method='umap', n_comp=10)
        np.savez(metrics_path, **results, layers=LAYERS, epochs=EPOCHS)

    plot_clustering_metrics(results)

    print('\nDone!')
