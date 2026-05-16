"""
Шаг 7: Оценка I(T;Y) через Classification AE + KSG.

Для каждого слоя × эпохи:
  1. Обучить AE с classification head на train (50000):
     act(N) → Z(16) → act'(N), Z → Y'(10)
  2. Проверить: acc(Z→Y) ≈ acc(T→Y) на test (10000)?
  3. I(Z;Y) = nonparam_mi_cd(Z, labels) — KSG в 16D пространстве
  4. Для сравнения: I_ind = Σ I(t_i; Y) (per-neuron, exact discrete)

Зависимости: driada (ClassificationLoss, KSG), torch, sklearn, numpy.
Данные: mango_data/Activity/SJ-SNN-50/
"""

import sys
import gc
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', message='.*lbfgs.*')
import numpy as np
from pathlib import Path

sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')

from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn

from driada.dim_reduction.flexible_ae import ModularAutoencoder
from driada.information.ksg import nonparam_mi_cd

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
N_CLASSES = 10
TRAIN_SUBSAMPLE = 10000  # подвыборка из train (1000/класс) для скорости

# AE hyperparameters
LATENT_DIM = 16
HIDDEN_DIM = 256
AE_EPOCHS = 50
AE_LR = 1e-3
AE_BATCH = 128
CLASS_WEIGHT = 1.0  # вес classification loss


def load_labels(split='test'):
    return np.load(DATA_ROOT / f'CIFAR {split} labels.npz')['arr_0']


def load_activity(epoch, layer, split='test'):
    epoch_dir = SNN_DIR / f'iter {epoch}' / split
    if split == 'train':
        fname = f'SJ-SNN act iter {epoch} {split} layer {layer}.npz'
    else:
        fname = f'SJ-SNN act iter {epoch} {split} layer {layer}.npy.npz'
    if layer == 'sn1':
        if split == 'train':
            fname = f'SJ-SNN act iter {epoch} {split} layer sn1.npz'
        else:
            fname = f'SJ-SNN act iter {epoch} {split} layer sn1.npy.npz'
    path = epoch_dir / fname
    if not path.exists():
        return None
    try:
        data = np.load(path)
    except Exception:
        return None
    key = 'arr_0' if 'arr_0' in data else 'a'
    arr = data[key]  # (T, n_images, N)
    return arr.sum(axis=0)  # (n_images, N)


def stratified_subsample(act, labels, n_total=TRAIN_SUBSAMPLE):
    """Стратифицированная подвыборка: n_total/n_classes из каждого класса."""
    n_per_class = n_total // N_CLASSES
    idx = []
    rng = np.random.RandomState(42)
    for c in range(N_CLASSES):
        mask = np.where(labels == c)[0]
        chosen = rng.choice(mask, size=min(n_per_class, len(mask)), replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return act[idx], labels[idx]


# ============================================================================
# Обучение AE + Classification head
# ============================================================================

def train_ae_classifier(act_train, labels_train, act_test, labels_test):
    """Обучить AE на train, вернуть латентные коды и accuracy на test.

    Returns:
        Z_test: (n_test, LATENT_DIM) — латентные коды test set
        acc_z: float — accuracy классификатора из Z на test
        recon_mse: float — средняя MSE реконструкции на test
    """
    N = act_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ModularAutoencoder(
        input_dim=N,
        latent_dim=LATENT_DIM,
        hidden_dim=min(HIDDEN_DIM, N * 2),
        loss_components=[
            {"name": "reconstruction", "weight": 1.0},
            {"name": "classification", "weight": CLASS_WEIGHT, "num_classes": N_CLASSES},
        ],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)

    train_tensor = torch.tensor(act_train, dtype=torch.float32).to(device)
    train_labels = torch.tensor(labels_train, dtype=torch.long).to(device)

    dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=AE_BATCH, shuffle=True)

    # Training on train set
    model.train()
    for ep in range(AE_EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            total_loss, _ = model.compute_loss(batch_x, labels=batch_y)
            total_loss.backward()
            optimizer.step()

    # Evaluate on test set
    model.eval()
    test_tensor = torch.tensor(act_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z_test = model.encode(test_tensor).cpu().numpy()
        recon = model(test_tensor).cpu().numpy()
        recon_mse = np.mean((act_test - recon) ** 2)

        cls_loss = model.losses[1]  # ClassificationLoss
        logits = cls_loss.classifier(model.encode(test_tensor))
        preds = logits.argmax(dim=1).cpu().numpy()
        acc_z = np.mean(preds == labels_test)

    return Z_test, acc_z, recon_mse


def gaussian_entropy(act):
    """H_gauss(T) = ½ Σ log(2πe λₖ), bits.

    Гауссова верхняя граница на H(T). λₖ — собственные значения ковариационной
    матрицы. Малые λₖ обрезаются (< 1e-10) чтобы избежать log(0).
    """
    cov = np.cov(act, rowvar=False)  # (N, N)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return 0.5 * np.sum(np.log2(2 * np.pi * np.e * eigenvalues))


def baseline_accuracy(act_train, labels_train, act_test, labels_test):
    """Accuracy логистической регрессии: train → test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(max_iter=500, solver='lbfgs', n_jobs=-1)
        lr.fit(act_train, labels_train)
        return lr.score(act_test, labels_test)


# ============================================================================
# Основной расчёт
# ============================================================================

def compute_all(labels_train, labels_test):
    n_layers = len(LAYERS)
    n_epochs = len(EPOCHS)

    I_latent = np.full((n_layers, n_epochs), np.nan)   # I(Z;Y) via KSG on test
    I_ind = np.full((n_layers, n_epochs), np.nan)       # Σ I(t_i; Y) per-neuron on test
    H_gauss = np.full((n_layers, n_epochs), np.nan)     # H_gauss(T) — Gaussian entropy
    acc_z = np.full((n_layers, n_epochs), np.nan)       # acc from Z on test
    acc_full = np.full((n_layers, n_epochs), np.nan)    # acc from full act (train→test)
    mse = np.full((n_layers, n_epochs), np.nan)         # reconstruction MSE on test

    for j, epoch in enumerate(EPOCHS):
        print(f'  Epoch {epoch}...', flush=True)
        for i, layer in enumerate(LAYERS):
            act_test = load_activity(epoch, layer, split='test')
            if act_test is None:
                continue

            # Per-neuron MI on test (exact discrete)
            mi_per_neuron = np.array([
                mutual_info_score(labels_test, act_test[:, k]) / LN2
                for k in range(act_test.shape[1])
            ])
            I_ind[i, j] = mi_per_neuron.sum()

            # Gaussian entropy H(T)
            H_gauss[i, j] = gaussian_entropy(act_test)

            # Load train data and subsample for speed
            act_train_full = load_activity(epoch, layer, split='train')
            if act_train_full is None:
                continue
            act_train, lab_train = stratified_subsample(
                act_train_full, labels_train)
            del act_train_full

            # Baseline accuracy (train → test)
            acc_full[i, j] = baseline_accuracy(act_train, lab_train,
                                                act_test, labels_test)

            # AE + Classification: train on train, evaluate on test
            try:
                Z_test, az, rm = train_ae_classifier(
                    act_train, lab_train, act_test, labels_test)
                acc_z[i, j] = az
                mse[i, j] = rm

                # I(Z; Y) via KSG on test set
                I_latent[i, j] = nonparam_mi_cd(Z_test, labels_test,
                                                 k=5, base=2.0)
            except Exception as e:
                print(f'    AE error {layer}: {e}')

            del act_test, act_train
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f'    done (epoch {epoch})')

    return I_latent, I_ind, H_gauss, acc_z, acc_full, mse


# ============================================================================
# Визуализация
# ============================================================================

def plot_latent_mi(I_latent, I_ind, acc_z, acc_full):
    """I(Z;Y) и контрольные метрики по эпохам."""
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
    cmap = plt.get_cmap('plasma')

    # --- Plot 1: I(Z;Y) vs I_ind ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    for col, (group_name, indices) in enumerate(groups):
        clrs = cmap(np.linspace(0.1, 0.9, len(indices)))

        # I(Z;Y)
        ax = make_beautiful(axes[0, col])
        for ci, idx in enumerate(indices):
            vals = I_latent[idx, mask_ep]
            valid = ~np.isnan(vals)
            if valid.any():
                ax.plot(xdata[valid], vals[valid], 'o-', c=clrs[ci],
                        lw=2, ms=4, label=SHORT[idx])
        ax.set_title(f'{group_name}: I(Z;Y) latent GCMI', fontsize=10)
        ax.set_ylabel('bits')
        ax.axhline(np.log2(N_CLASSES), ls='--', c='grey', alpha=0.5, label='H(Y)')
        ax.legend(fontsize=7)

        # I_ind
        ax2 = make_beautiful(axes[1, col])
        for ci, idx in enumerate(indices):
            vals = I_ind[idx, mask_ep]
            valid = ~np.isnan(vals)
            if valid.any():
                ax2.plot(xdata[valid], vals[valid], 'o-', c=clrs[ci],
                         lw=2, ms=4, label=SHORT[idx])
        ax2.set_title(f'{group_name}: I_ind (sum per-neuron)', fontsize=10)
        ax2.set_xlabel(r'$\log_{10}(\mathrm{epoch}+1)$')
        ax2.set_ylabel('bits')
        ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'latent_mi_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "latent_mi_curves.png"}')

    # --- Plot 2: accuracy control ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    for col, (group_name, indices) in enumerate(groups):
        clrs = cmap(np.linspace(0.1, 0.9, len(indices)))
        ax = make_beautiful(axes[col])
        for ci, idx in enumerate(indices):
            az = acc_z[idx, mask_ep]
            af = acc_full[idx, mask_ep]
            valid = ~np.isnan(az) & ~np.isnan(af)
            if valid.any():
                ax.plot(xdata[valid], af[valid], 'o--', c=clrs[ci],
                        lw=1.5, ms=3, alpha=0.5)
                ax.plot(xdata[valid], az[valid], 's-', c=clrs[ci],
                        lw=2, ms=4, label=SHORT[idx])
        ax.set_title(f'{group_name}: acc (-- full, - latent)', fontsize=10)
        ax.set_xlabel(r'$\log_{10}(\mathrm{epoch}+1)$')
        ax.set_ylabel('accuracy')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'latent_mi_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "latent_mi_accuracy.png"}')


def plot_by_layer(I_latent, I_ind, acc_z, acc_full):
    """I(Z;Y), I_ind, accuracy по слоям для разных эпох."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
    from driada.intense.visual import make_beautiful

    SHORT = []
    for l in LAYERS:
        if l == 'sn1':
            SHORT.append('sn1')
        else:
            parts = l.split('.')
            block = parts[0].replace('layer', 'L')
            SHORT.append(f'{block}.{parts[1]}.{parts[2][-1]}')

    SHOW_EPOCHS = [5, 20, 50, 100, 200, 400, 700, 900]
    epoch_indices = [EPOCHS.index(e) for e in SHOW_EPOCHS if e in EPOCHS]
    show_ep = [EPOCHS[j] for j in epoch_indices]

    cmap = plt.get_cmap('coolwarm')
    clrs = cmap(np.linspace(0, 1, len(epoch_indices)))
    x = np.arange(len(LAYERS))

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # --- I(Z;Y) ---
    ax = make_beautiful(axes[0, 0])
    for ci, j in enumerate(epoch_indices):
        vals = I_latent[:, j]
        mask = ~np.isnan(vals)
        if mask.any():
            ax.plot(x[mask], vals[mask], 'o-', c=clrs[ci], lw=2.5, ms=6,
                    label=f'эпоха {show_ep[ci]}', alpha=0.85)
    ax.axhline(np.log2(N_CLASSES), ls='--', c='grey', alpha=0.5, label='H(Y)')
    for sep in [0.5, 4.5, 8.5, 12.5]:
        ax.axvline(sep, c='grey', ls=':', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('бит')
    ax.set_title('I(Z; Y) — латентная ВИ (KSG)')
    ax.legend(fontsize=8, ncol=4, loc='upper left')

    # --- I_ind ---
    ax = make_beautiful(axes[0, 1])
    for ci, j in enumerate(epoch_indices):
        vals = I_ind[:, j]
        mask = ~np.isnan(vals)
        if mask.any():
            ax.plot(x[mask], vals[mask], 'o-', c=clrs[ci], lw=2.5, ms=6,
                    label=f'эпоха {show_ep[ci]}', alpha=0.85)
    for sep in [0.5, 4.5, 8.5, 12.5]:
        ax.axvline(sep, c='grey', ls=':', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('бит')
    ax.set_title(r'$I_{ind} = \Sigma\, I(t_i;\, Y)$ — сумма по нейронам')
    ax.legend(fontsize=8, ncol=4, loc='upper left')

    # --- Accuracy (latent) ---
    ax = make_beautiful(axes[1, 0])
    for ci, j in enumerate(epoch_indices):
        vals = acc_z[:, j]
        mask = ~np.isnan(vals)
        if mask.any():
            ax.plot(x[mask], vals[mask], 'o-', c=clrs[ci], lw=2.5, ms=6,
                    label=f'эпоха {show_ep[ci]}', alpha=0.85)
    for sep in [0.5, 4.5, 8.5, 12.5]:
        ax.axvline(sep, c='grey', ls=':', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('точность')
    ax.set_title('acc(Z → Y) — латентный классификатор')
    ax.legend(fontsize=8, ncol=4, loc='upper left')

    # --- Accuracy ratio ---
    ax = make_beautiful(axes[1, 1])
    for ci, j in enumerate(epoch_indices):
        az = acc_z[:, j]
        af = acc_full[:, j]
        ratio = np.where((af > 0) & ~np.isnan(az) & ~np.isnan(af), az / af, np.nan)
        mask = ~np.isnan(ratio)
        if mask.any():
            ax.plot(x[mask], ratio[mask], 'o-', c=clrs[ci], lw=2.5, ms=6,
                    label=f'эпоха {show_ep[ci]}', alpha=0.85)
    ax.axhline(1.0, ls='--', c='grey', alpha=0.5)
    for sep in [0.5, 4.5, 8.5, 12.5]:
        ax.axvline(sep, c='grey', ls=':', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('отношение')
    ax.set_title('acc(Z→Y) / acc(T→Y)')
    ax.legend(fontsize=8, ncol=4, loc='upper left')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'latent_mi_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "latent_mi_by_layer.png"}')


def plot_info_plane(I_latent, H_gauss):
    """Information plane: H_gauss(T) vs I(Z;Y) — траектории слоёв по эпохам."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
    from driada.intense.visual import make_beautiful

    SHORT = []
    for l in LAYERS:
        if l == 'sn1':
            SHORT.append('sn1')
        else:
            parts = l.split('.')
            block = parts[0].replace('layer', 'L')
            SHORT.append(f'{block}.{parts[1]}.{parts[2][-1]}')

    groups = [
        ('sn1 + layer1', [0, 1, 2, 3, 4]),
        ('layer2', [5, 6, 7, 8]),
        ('layer3', [9, 10, 11, 12]),
        ('layer4', [13, 14, 15, 16]),
    ]

    # Подмножество эпох для стрелок
    SHOW_EPOCHS = [0, 5, 10, 20, 50, 100, 200, 400, 900]
    ep_idx = [EPOCHS.index(e) for e in SHOW_EPOCHS if e in EPOCHS]

    cmap = plt.get_cmap('plasma')

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    for col, (group_name, indices) in enumerate(groups):
        ax = make_beautiful(axes[col])
        clrs = cmap(np.linspace(0.1, 0.9, len(indices)))

        for ci, idx in enumerate(indices):
            hx = H_gauss[idx, :]
            iy = I_latent[idx, :]
            valid = ~np.isnan(hx) & ~np.isnan(iy)

            if not valid.any():
                continue

            # Полная траектория — тонкая линия
            ax.plot(hx[valid], iy[valid], '-', c=clrs[ci], lw=1, alpha=0.4)

            # Точки для выбранных эпох
            for j in ep_idx:
                if j < len(hx) and valid[j]:
                    ax.plot(hx[j], iy[j], 'o', c=clrs[ci], ms=5, alpha=0.8)

            # Стрелка от первой к последней валидной точке
            first = np.where(valid)[0][0]
            last = np.where(valid)[0][-1]
            ax.annotate('', xy=(hx[last], iy[last]),
                        xytext=(hx[first], iy[first]),
                        arrowprops=dict(arrowstyle='->', color=clrs[ci],
                                        lw=2, alpha=0.7))
            # Подпись слоя у последней точки
            ax.annotate(SHORT[idx], (hx[last], iy[last]),
                        fontsize=7, color=clrs[ci], fontweight='bold',
                        xytext=(4, 4), textcoords='offset points')

        ax.axhline(np.log2(N_CLASSES), ls='--', c='grey', alpha=0.5, label='H(Y)')
        ax.set_xlabel(r'$H_{gauss}(T)$ (бит)')
        ax.set_ylabel(r'$I(Z; Y)$ (бит)')
        ax.set_title(f'{group_name}')
        ax.legend(fontsize=8)

    plt.suptitle('Информационная плоскость: H(T) vs I(T;Y)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'information_plane.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "information_plane.png"}')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    labels_train = load_labels('train')
    labels_test = load_labels('test')
    print(f'Epochs: {EPOCHS}')
    print(f'Layers: {len(LAYERS)}')
    print(f'Train: {labels_train.shape}, Test: {labels_test.shape}')

    npz_path = OUT_DIR / 'latent_mi.npz'
    if npz_path.exists():
        print('Loading precomputed results...')
        data = np.load(npz_path)
        I_latent = data['I_latent']
        I_ind = data['I_ind']
        acc_z_arr = data['acc_z']
        acc_full_arr = data['acc_full']
        mse_arr = data['mse']
        H_gauss_arr = data['H_gauss'] if 'H_gauss' in data else None
    else:
        print('\n=== Computing latent MI ===')
        I_latent, I_ind, H_gauss_arr, acc_z_arr, acc_full_arr, mse_arr = compute_all(
            labels_train, labels_test)
        np.savez(npz_path,
                 I_latent=I_latent, I_ind=I_ind, H_gauss=H_gauss_arr,
                 acc_z=acc_z_arr, acc_full=acc_full_arr, mse=mse_arr,
                 layers=LAYERS, epochs=EPOCHS)
        print(f'Saved: {npz_path}')

    # Compute H_gauss if not in cache
    if H_gauss_arr is None:
        print('\n=== Computing H_gauss(T) ===')
        H_gauss_arr = np.full((len(LAYERS), len(EPOCHS)), np.nan)
        for j, epoch in enumerate(EPOCHS):
            print(f'  Epoch {epoch}...', end=' ', flush=True)
            for i, layer in enumerate(LAYERS):
                act = load_activity(epoch, layer, split='test')
                if act is not None:
                    H_gauss_arr[i, j] = gaussian_entropy(act)
            print('done')
        # Re-save with H_gauss
        np.savez(npz_path,
                 I_latent=I_latent, I_ind=I_ind, H_gauss=H_gauss_arr,
                 acc_z=acc_z_arr, acc_full=acc_full_arr, mse=mse_arr,
                 layers=LAYERS, epochs=EPOCHS)
        print(f'Saved: {npz_path}')

    print('\n=== Plotting ===')
    plot_latent_mi(I_latent, I_ind, acc_z_arr, acc_full_arr)
    plot_by_layer(I_latent, I_ind, acc_z_arr, acc_full_arr)
    plot_info_plane(I_latent, H_gauss_arr)

    # Summary
    print('\n=== Summary ===')
    mask_ep = np.array(EPOCHS) > 0
    print(f'{"Layer":16s} {"I(Z;Y) final":>12s} {"I_ind final":>12s} '
          f'{"acc_Z":>8s} {"acc_full":>8s} {"ratio":>8s}')
    print('-' * 70)
    SHORT = []
    for l in LAYERS:
        if l == 'sn1':
            SHORT.append('sn1')
        else:
            parts = l.split('.')
            block = parts[0].replace('layer', 'L')
            SHORT.append(f'{block}.{parts[1]}.{parts[2][-1]}')

    for i, layer in enumerate(LAYERS):
        il = I_latent[i, -1]
        ii = I_ind[i, -1]
        az = acc_z_arr[i, -1]
        af = acc_full_arr[i, -1]
        ratio = az / af if af > 0 and not np.isnan(az) else np.nan
        print(f'{SHORT[i]:16s} {il:12.3f} {ii:12.3f} '
              f'{az:8.3f} {af:8.3f} {ratio:8.3f}')

    print('\nDone!')
