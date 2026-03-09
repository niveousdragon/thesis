"""
Шаг 5: Взаимная информация I(T;Y) — активность слоя vs метки классов.

Три уровня:
  Level 0: Per-neuron MI (exact discrete) → I_ind = Σ I(t_i; Y)
  Level 1: Gaussian multivariate MI → I_gauss via mi_model_gd
  Level 2: LDA spectral decomposition → I_k = 0.5·log₂(1+λ_k)

Данные: mango_data/Activity/SJ-SNN-50/
"""

import sys
import gc
import numpy as np
from pathlib import Path
from sklearn.metrics import mutual_info_score
from scipy.linalg import eigh

sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
from driada.information.gcmi import mi_model_gd

# ============================================================================
# Параметры
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

LN2 = np.log(2)


def load_labels():
    return np.load(DATA_ROOT / 'CIFAR test labels.npz')['arr_0']


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
# Level 0: Per-neuron MI (exact discrete)
# ============================================================================

def compute_per_neuron_mi(act, labels):
    """I(t_i; Y) для каждого нейрона, exact discrete, в битах."""
    N = act.shape[1]
    mi = np.zeros(N)
    for i in range(N):
        mi[i] = mutual_info_score(labels, act[:, i]) / LN2
    return mi


# ============================================================================
# Level 1: Gaussian multivariate MI
# ============================================================================

def compute_gauss_mi(act, labels):
    """I_gauss(T; Y) через mi_model_gd (DRIADA)."""
    return mi_model_gd(act.T, labels)


# ============================================================================
# Level 2: LDA spectral decomposition
# ============================================================================

def compute_lda_spectrum(act, labels, n_classes=10):
    """Дискриминантный спектр: eigenvalues λ_k of Σ_W⁻¹·Σ_B.

    Returns:
        lambda_k: (K-1,) — дискриминантные отношения, убывающие
        I_k: (K-1,) — информация по каждой оси, bits
        I_lda: float — суммарная MI (гомоскедастичная аппроксимация)
        ed_disc: float — эффективное число дискриминантных осей (PR от I_k)
    """
    n_samples, N = act.shape
    K = n_classes

    # Between-class scatter
    mu = act.mean(axis=0)
    Sb = np.zeros((N, N))
    Sw = np.zeros((N, N))

    for c in range(K):
        mask = labels == c
        Xc = act[mask]
        nc = Xc.shape[0]
        if nc < 2:
            continue
        mu_c = Xc.mean(axis=0)
        diff = (mu_c - mu).reshape(-1, 1)
        Sb += nc * (diff @ diff.T)
        Xc_centered = Xc - mu_c
        Sw += Xc_centered.T @ Xc_centered

    Sb /= n_samples
    Sw /= (n_samples - K)

    # Regularize Sw (ridge, alpha = 1e-4 * trace/N)
    alpha = 1e-4 * np.trace(Sw) / N
    Sw_reg = Sw + alpha * np.eye(N)

    # Generalized eigenvalue problem: Sb v = lambda Sw v
    # Get top K-1 eigenvalues
    n_components = min(K - 1, N)
    try:
        lambda_k, _ = eigh(Sb, Sw_reg,
                           subset_by_index=[N - n_components, N - 1])
        lambda_k = lambda_k[::-1]  # descending
        lambda_k = np.maximum(lambda_k, 0)  # clip negatives
    except Exception as e:
        print(f'  LDA eigh error: {e}')
        lambda_k = np.full(n_components, np.nan)

    # MI per axis
    I_k = 0.5 * np.log2(1 + lambda_k)
    I_lda = np.nansum(I_k)

    # Effective discriminant dimensionality (PR)
    I_k_pos = I_k[I_k > 0]
    if len(I_k_pos) > 0 and np.sum(I_k_pos) > 0:
        ed_disc = np.sum(I_k_pos) ** 2 / np.sum(I_k_pos ** 2)
    else:
        ed_disc = 0.0

    return lambda_k, I_k, I_lda, ed_disc


# ============================================================================
# Основной расчёт
# ============================================================================

def compute_all(labels):
    n_layers = len(LAYERS)
    n_epochs = len(EPOCHS)

    I_ind = np.full((n_layers, n_epochs), np.nan)
    I_gauss = np.full((n_layers, n_epochs), np.nan)
    I_lda = np.full((n_layers, n_epochs), np.nan)
    ed_disc = np.full((n_layers, n_epochs), np.nan)
    lda_spectrum = np.full((n_layers, n_epochs, 9), np.nan)

    for j, epoch in enumerate(EPOCHS):
        print(f'  Epoch {epoch}...', end=' ', flush=True)
        for i, layer in enumerate(LAYERS):
            act = load_activity(epoch, layer)
            if act is None:
                continue

            # Level 0: per-neuron MI
            mi_per_neuron = compute_per_neuron_mi(act, labels)
            I_ind[i, j] = mi_per_neuron.sum()

            # Level 1: Gaussian multivariate MI
            try:
                I_gauss[i, j] = compute_gauss_mi(act, labels)
            except Exception as e:
                print(f'gauss error {layer}: {e}')

            # Level 2: LDA spectral
            try:
                lam, ik, il, ed = compute_lda_spectrum(act, labels)
                I_lda[i, j] = il
                ed_disc[i, j] = ed
                n_comp = min(9, len(lam))
                lda_spectrum[i, j, :n_comp] = lam[:n_comp]
            except Exception as e:
                print(f'lda error {layer}: {e}')

            del act
            gc.collect()
        print('done')

    return I_ind, I_gauss, I_lda, ed_disc, lda_spectrum


# ============================================================================
# Визуализация
# ============================================================================

def plot_mi_curves(I_ind, I_gauss, I_lda):
    """I(T;Y) по эпохам для групп слоёв."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
    from driada.intense.visual import make_beautiful

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

    fig, axes = plt.subplots(3, 4, figsize=(24, 15))
    cmap = plt.get_cmap('plasma')

    titles_row = ['I_ind (sum per-neuron)', 'I_gauss (multivariate)', 'I_LDA (spectral)']
    data_row = [I_ind, I_gauss, I_lda]

    for row, (data, row_title) in enumerate(zip(data_row, titles_row)):
        for col, (group_name, indices) in enumerate(groups):
            clrs = cmap(np.linspace(0.1, 0.9, len(indices)))
            ax = make_beautiful(axes[row, col])
            for ci, idx in enumerate(indices):
                vals = data[idx, mask_ep]
                valid = ~np.isnan(vals)
                if valid.any():
                    ax.plot(xdata[valid], vals[valid], 'o-', c=clrs[ci],
                            lw=2, ms=4, label=SHORT[idx])
            ax.set_title(f'{group_name}: {row_title}', fontsize=10)
            ax.set_xlabel(r'$\log_{10}(\mathrm{epoch}+1)$')
            ax.set_ylabel('bits')
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'information_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "information_curves.png"}')


def plot_synergy_redundancy(I_ind, I_gauss):
    """I_gauss / I_ind по эпохам."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
    from driada.intense.visual import make_beautiful

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

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    cmap = plt.get_cmap('plasma')

    for col, (group_name, indices) in enumerate(groups):
        clrs = cmap(np.linspace(0.1, 0.9, len(indices)))
        ax = make_beautiful(axes[col])
        ax.axhline(1.0, c='grey', ls='--', alpha=0.5, label='independent')
        for ci, idx in enumerate(indices):
            ratio = I_gauss[idx, mask_ep] / I_ind[idx, mask_ep]
            valid = ~np.isnan(ratio) & ~np.isinf(ratio)
            if valid.any():
                ax.plot(xdata[valid], ratio[valid], 'o-', c=clrs[ci],
                        lw=2, ms=4, label=SHORT[idx])
        ax.set_title(f'{group_name}: I_gauss / I_ind', fontsize=11)
        ax.set_xlabel(r'$\log_{10}(\mathrm{epoch}+1)$')
        ax.set_ylabel('ratio (>1 = synergy, <1 = redundancy)')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'synergy_redundancy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "synergy_redundancy.png"}')


def plot_information_plane(I_gauss):
    """Информационная плоскость: (ВР, I_gauss) траектории."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
    from driada.intense.visual import make_beautiful

    dim_data = np.load(OUT_DIR / 'dimensionality.npz', allow_pickle=True)
    id_nn = dim_data['id_nn']

    SHORT = []
    for l in LAYERS:
        if l == 'sn1':
            SHORT.append('sn1')
        else:
            parts = l.split('.')
            block = parts[0].replace('layer', 'L')
            SHORT.append(f'{block}.{parts[1]}.{parts[2][-1]}')

    # Показать только sn2 слои (один на блок) + sn1, чтобы не загромождать
    show_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16]  # sn1 + все sn2
    mask_ep = np.array(EPOCHS) > 0

    cmap = plt.get_cmap('viridis')
    clrs = cmap(np.linspace(0.1, 0.9, len(show_indices)))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax = make_beautiful(ax)

    for ci, idx in enumerate(show_indices):
        x = id_nn[idx, mask_ep]
        y = I_gauss[idx, mask_ep]
        valid = ~np.isnan(x) & ~np.isnan(y)
        if valid.sum() < 2:
            continue
        ax.plot(x[valid], y[valid], 'o-', c=clrs[ci], lw=2, ms=5,
                label=SHORT[idx], alpha=0.8)
        # Стрелка направления (от первой к последней точке)
        if valid.sum() >= 3:
            ax.annotate('', xy=(x[valid][-1], y[valid][-1]),
                        xytext=(x[valid][-2], y[valid][-2]),
                        arrowprops=dict(arrowstyle='->', color=clrs[ci], lw=2))
        # Подпись начальной эпохи
        ax.text(x[valid][0], y[valid][0], f' ep{EPOCHS[1]}',
                fontsize=7, alpha=0.6)

    ax.set_xlabel('Intrinsic dimension (2-NN)', fontsize=12)
    ax.set_ylabel('I(T; Y) bits [Gaussian]', fontsize=12)
    ax.set_title('Information plane: ВР vs I(T;Y)', fontsize=13)
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'information_plane.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "information_plane.png"}')


def plot_lda_spectrum_examples(lda_spectrum):
    """LDA спектр для ключевых слоёв/эпох."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
    from driada.intense.visual import make_beautiful

    show_layers = [0, 8, 12, 16]  # sn1, layer2.1.sn2, layer3.1.sn2, layer4.1.sn2
    show_epochs_vals = [5, 30, 100, 900]
    show_epoch_idx = [EPOCHS.index(e) for e in show_epochs_vals]

    SHORT = ['sn1', 'L2.1.2', 'L3.1.2', 'L4.1.2']

    fig, axes = plt.subplots(len(show_layers), len(show_epochs_vals),
                              figsize=(16, 12))

    for row, (li, sname) in enumerate(zip(show_layers, SHORT)):
        for col, (ei, ep) in enumerate(zip(show_epoch_idx, show_epochs_vals)):
            ax = make_beautiful(axes[row, col])
            spectrum = lda_spectrum[li, ei, :]
            valid = ~np.isnan(spectrum)
            if valid.any():
                I_k = 0.5 * np.log2(1 + spectrum[valid])
                ax.bar(range(len(I_k)), I_k, color='steelblue', alpha=0.8)
                ax.set_ylim(0, max(I_k.max() * 1.2, 0.1))
            ax.set_title(f'{sname}, ep {ep}', fontsize=10)
            if row == len(show_layers) - 1:
                ax.set_xlabel('Discriminant axis')
            if col == 0:
                ax.set_ylabel('I_k (bits)')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'lda_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "lda_spectrum.png"}')


def plot_ed_disc_vs_er(ed_disc):
    """ED_disc и ЭР по эпохам."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
    from driada.intense.visual import make_beautiful

    dim_data = np.load(OUT_DIR / 'dimensionality.npz', allow_pickle=True)
    ed = dim_data['ed']

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

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    cmap = plt.get_cmap('plasma')

    for col, (group_name, indices) in enumerate(groups):
        clrs = cmap(np.linspace(0.1, 0.9, len(indices)))
        ax = make_beautiful(axes[col])
        for ci, idx in enumerate(indices):
            ratio = ed_disc[idx, mask_ep] / ed[idx, mask_ep]
            valid = ~np.isnan(ratio) & ~np.isinf(ratio)
            if valid.any():
                ax.plot(xdata[valid], ratio[valid], 'o-', c=clrs[ci],
                        lw=2, ms=4, label=SHORT[idx])
        ax.set_title(f'{group_name}: ED_disc / ЭР', fontsize=11)
        ax.set_xlabel(r'$\log_{10}(\mathrm{epoch}+1)$')
        ax.set_ylabel('fraction class-relevant')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'ed_disc_ratio.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "ed_disc_ratio.png"}')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f'Epochs: {EPOCHS}')
    print(f'Layers: {len(LAYERS)}')

    labels = load_labels()
    print(f'Labels: {labels.shape}, classes: {np.unique(labels)}')

    npz_path = OUT_DIR / 'information.npz'
    if npz_path.exists():
        print('Loading precomputed information...')
        data = np.load(npz_path)
        I_ind = data['I_ind']
        I_gauss = data['I_gauss']
        I_lda = data['I_lda']
        ed_disc = data['ed_disc']
        lda_spectrum = data['lda_spectrum']
    else:
        print('\n=== Computing mutual information ===')
        I_ind, I_gauss, I_lda, ed_disc, lda_spectrum = compute_all(labels)
        np.savez(npz_path,
                 I_ind=I_ind, I_gauss=I_gauss, I_lda=I_lda,
                 ed_disc=ed_disc, lda_spectrum=lda_spectrum,
                 layers=LAYERS, epochs=EPOCHS)
        print(f'Saved: {npz_path}')

    print('\n=== Plotting ===')
    plot_mi_curves(I_ind, I_gauss, I_lda)
    plot_synergy_redundancy(I_ind, I_gauss)
    plot_information_plane(I_gauss)
    plot_lda_spectrum_examples(lda_spectrum)
    plot_ed_disc_vs_er(ed_disc)

    print('\nDone!')
