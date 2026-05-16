"""
Шаг 7c: I(T;Y) через Classification AE с MLP head + Fano sandwich bound.

Отличия от 07_latent_mi.py:
  - ClassificationLoss с hidden_dim=64 (2-слойный MLP вместо Linear)
  - Fano sandwich: I(Z;Y) ≤ I(T;Y) ≤ min(H(Y), I(Z;Y) + h(Pe) + Pe·log₂(|Y|-1))
  - Redundancy–synergy ratio: I(Z;Y) / I_ind

Зависимости: driada (ClassificationLoss с hidden_dim, KSG), torch, sklearn, numpy.
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
TRAIN_SUBSAMPLE = 10000

# AE hyperparameters
LATENT_DIM = 16
HIDDEN_DIM = 256
AE_EPOCHS = 50
AE_LR = 1e-3
AE_BATCH = 128
CLASS_WEIGHT = 1.0
CLASS_HIDDEN = 64  # MLP head hidden dim


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
# AE + MLP Classification head
# ============================================================================

def train_ae_classifier(act_train, labels_train, act_test, labels_test, seed=42):
    """Обучить AE с MLP classification head на train, вернуть латентные коды и accuracy на test."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = act_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ModularAutoencoder(
        input_dim=N,
        latent_dim=LATENT_DIM,
        hidden_dim=min(HIDDEN_DIM, N * 2),
        loss_components=[
            {"name": "reconstruction", "weight": 1.0},
            {"name": "classification", "weight": CLASS_WEIGHT,
             "num_classes": N_CLASSES, "hidden_dim": CLASS_HIDDEN},
        ],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)

    train_tensor = torch.tensor(act_train, dtype=torch.float32).to(device)
    train_labels = torch.tensor(labels_train, dtype=torch.long).to(device)

    dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=AE_BATCH, shuffle=True)

    model.train()
    for ep in range(AE_EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            total_loss, _ = model.compute_loss(batch_x, labels=batch_y)
            total_loss.backward()
            optimizer.step()

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
    """H_gauss(T) = ½ Σ log(2πe λₖ), bits."""
    cov = np.cov(act, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return 0.5 * np.sum(np.log2(2 * np.pi * np.e * eigenvalues))


def baseline_accuracy(act_train, labels_train, act_test, labels_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(max_iter=500, solver='lbfgs', n_jobs=-1)
        lr.fit(act_train, labels_train)
        return lr.score(act_test, labels_test)


def fano_bound(acc, n_classes=N_CLASSES):
    """Fano lower bound on MI: H(Y) - h(Pe) - Pe·log₂(|Y|-1)."""
    H_Y = np.log2(n_classes)
    pe = 1.0 - acc
    pe = np.clip(pe, 1e-10, 1 - 1e-10)
    h_pe = -pe * np.log2(pe) - (1 - pe) * np.log2(1 - pe)
    return H_Y - h_pe - pe * np.log2(n_classes - 1)


def fano_gap_upper(acc, n_classes=N_CLASSES):
    """Upper bound on gap I(T;Y) - I(Z;Y) via Fano: h(Pe^Z) + Pe^Z·log₂(|Y|-1)."""
    pe = 1.0 - acc
    pe = np.clip(pe, 1e-10, 1 - 1e-10)
    h_pe = -pe * np.log2(pe) - (1 - pe) * np.log2(1 - pe)
    return h_pe + pe * np.log2(n_classes - 1)


# ============================================================================
# Основной расчёт
# ============================================================================

def compute_all(labels_train, labels_test):
    n_layers = len(LAYERS)
    n_epochs = len(EPOCHS)

    I_latent = np.full((n_layers, n_epochs), np.nan)
    I_ind = np.full((n_layers, n_epochs), np.nan)
    H_gauss = np.full((n_layers, n_epochs), np.nan)
    acc_z = np.full((n_layers, n_epochs), np.nan)
    acc_full = np.full((n_layers, n_epochs), np.nan)
    mse = np.full((n_layers, n_epochs), np.nan)

    for j, epoch in enumerate(EPOCHS):
        print(f'  Epoch {epoch}...', flush=True)
        for i, layer in enumerate(LAYERS):
            act_test = load_activity(epoch, layer, split='test')
            if act_test is None:
                continue

            mi_per_neuron = np.array([
                mutual_info_score(labels_test, act_test[:, k]) / LN2
                for k in range(act_test.shape[1])
            ])
            I_ind[i, j] = mi_per_neuron.sum()
            H_gauss[i, j] = gaussian_entropy(act_test)

            act_train_full = load_activity(epoch, layer, split='train')
            if act_train_full is None:
                continue
            act_train, lab_train = stratified_subsample(
                act_train_full, labels_train)
            del act_train_full

            acc_full[i, j] = baseline_accuracy(act_train, lab_train,
                                                act_test, labels_test)

            try:
                Z_test, az, rm = train_ae_classifier(
                    act_train, lab_train, act_test, labels_test)
                acc_z[i, j] = az
                mse[i, j] = rm

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

def short_names():
    SHORT = []
    for l in LAYERS:
        if l == 'sn1':
            SHORT.append('sn1')
        else:
            parts = l.split('.')
            block = parts[0].replace('layer', 'L')
            SHORT.append(f'{block}.{parts[1]}.{parts[2][-1]}')
    return SHORT


def plot_info_plane(I_latent, H_gauss):
    """Information plane: H_gauss(T) vs I(Z;Y)."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
    from driada.intense.visual import make_beautiful

    SHORT = short_names()
    groups = [
        ('sn1 + layer1', [0, 1, 2, 3, 4]),
        ('layer2', [5, 6, 7, 8]),
        ('layer3', [9, 10, 11, 12]),
        ('layer4', [13, 14, 15, 16]),
    ]

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
            ax.plot(hx[valid], iy[valid], '-', c=clrs[ci], lw=1, alpha=0.4)
            for j in ep_idx:
                if j < len(hx) and valid[j]:
                    ax.plot(hx[j], iy[j], 'o', c=clrs[ci], ms=5, alpha=0.8)
            first = np.where(valid)[0][0]
            last = np.where(valid)[0][-1]
            ax.annotate('', xy=(hx[last], iy[last]),
                        xytext=(hx[first], iy[first]),
                        arrowprops=dict(arrowstyle='->', color=clrs[ci],
                                        lw=2, alpha=0.7))
            ax.annotate(SHORT[idx], (hx[last], iy[last]),
                        fontsize=7, color=clrs[ci], fontweight='bold',
                        xytext=(4, 4), textcoords='offset points')

        ax.axhline(np.log2(N_CLASSES), ls='--', c='grey', alpha=0.5, label='H(Y)')
        ax.set_xlabel(r'$H_{gauss}(T)$ (бит)')
        ax.set_ylabel(r'$I(Z; Y)$ (бит)')
        ax.set_title(f'{group_name}')
        ax.legend(fontsize=8)

    plt.suptitle('Информационная плоскость: H(T) vs I(T;Y) [MLP head]', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'information_plane_mlp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "information_plane_mlp.png"}')


def plot_by_layer(I_latent, I_ind, acc_z, acc_full):
    """I(Z;Y), I_ind, accuracy, Fano sandwich по слоям."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
    from driada.intense.visual import make_beautiful

    SHORT = short_names()
    H_Y = np.log2(N_CLASSES)

    SHOW_EPOCHS = [5, 20, 50, 100, 200, 400, 700, 900]
    epoch_indices = [EPOCHS.index(e) for e in SHOW_EPOCHS if e in EPOCHS]
    show_ep = [EPOCHS[j] for j in epoch_indices]

    cmap = plt.get_cmap('coolwarm')
    clrs = cmap(np.linspace(0, 1, len(epoch_indices)))
    x = np.arange(len(LAYERS))

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # --- I(Z;Y) with Fano sandwich ---
    ax = make_beautiful(axes[0, 0])
    for ci, j in enumerate(epoch_indices):
        vals = I_latent[:, j]
        mask = ~np.isnan(vals)
        if mask.any():
            ax.plot(x[mask], vals[mask], 'o-', c=clrs[ci], lw=2.5, ms=6,
                    label=f'эпоха {show_ep[ci]}', alpha=0.85)
            # Fano upper bound
            az = acc_z[:, j]
            upper = np.minimum(H_Y, vals + fano_gap_upper(az))
            ax.fill_between(x[mask], vals[mask], upper[mask],
                            color=clrs[ci], alpha=0.1)
    ax.axhline(H_Y, ls='--', c='grey', alpha=0.5, label='H(Y)')
    for sep in [0.5, 4.5, 8.5, 12.5]:
        ax.axvline(sep, c='grey', ls=':', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('бит')
    ax.set_title('I(Z; Y) + Fano sandwich')
    ax.legend(fontsize=8, ncol=4, loc='upper left')

    # --- Redundancy–synergy ratio ---
    ax = make_beautiful(axes[0, 1])
    for ci, j in enumerate(epoch_indices):
        il = I_latent[:, j]
        ii = I_ind[:, j]
        ratio = np.where((ii > 0) & ~np.isnan(il), il / ii, np.nan)
        mask = ~np.isnan(ratio)
        if mask.any():
            ax.plot(x[mask], ratio[mask], 'o-', c=clrs[ci], lw=2.5, ms=6,
                    label=f'эпоха {show_ep[ci]}', alpha=0.85)
    ax.axhline(1.0, ls='--', c='grey', alpha=0.5, label='independent')
    for sep in [0.5, 4.5, 8.5, 12.5]:
        ax.axvline(sep, c='grey', ls=':', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('I(Z;Y) / I_ind')
    ax.set_title('Redundancy–synergy ratio')
    ax.legend(fontsize=8, ncol=4, loc='upper left')

    # --- Accuracy (latent MLP vs full LogReg) ---
    ax = make_beautiful(axes[1, 0])
    for ci, j in enumerate(epoch_indices):
        az = acc_z[:, j]
        af = acc_full[:, j]
        mask = ~np.isnan(az) & ~np.isnan(af)
        if mask.any():
            ax.plot(x[mask], af[mask], 'o--', c=clrs[ci], lw=1.5, ms=3, alpha=0.5)
            ax.plot(x[mask], az[mask], 's-', c=clrs[ci], lw=2, ms=4,
                    label=SHORT[0] if ci == 0 else f'эпоха {show_ep[ci]}')
    for sep in [0.5, 4.5, 8.5, 12.5]:
        ax.axvline(sep, c='grey', ls=':', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('точность')
    ax.set_title('acc(Z→Y) MLP [—] vs acc(T→Y) LogReg [--]')
    ax.legend(fontsize=8, ncol=4, loc='upper left')

    # --- Fano gap ---
    ax = make_beautiful(axes[1, 1])
    for ci, j in enumerate(epoch_indices):
        az = acc_z[:, j]
        gap = fano_gap_upper(az)
        mask = ~np.isnan(az)
        if mask.any():
            ax.plot(x[mask], gap[mask], 'o-', c=clrs[ci], lw=2.5, ms=6,
                    label=f'эпоха {show_ep[ci]}', alpha=0.85)
    for sep in [0.5, 4.5, 8.5, 12.5]:
        ax.axvline(sep, c='grey', ls=':', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('бит')
    ax.set_title('Fano gap upper bound')
    ax.legend(fontsize=8, ncol=4, loc='upper left')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'latent_mi_by_layer_mlp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "latent_mi_by_layer_mlp.png"}')


def plot_redundancy_synergy_dynamics(I_latent, I_ind):
    """Redundancy–synergy ratio по эпохам для ключевых слоёв."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')
    from driada.intense.visual import make_beautiful

    key_layers = {
        'sn1': 0,
        'layer1.1.sn2': 4,
        'layer2.1.sn2': 8,
        'layer3.1.sn2': 12,
        'layer4.1.sn2': 16,
    }

    mask_ep = np.array(EPOCHS) > 0
    xdata = np.log10(np.array(EPOCHS)[mask_ep] + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = make_beautiful(ax)

    cmap = plt.get_cmap('viridis')
    clrs = cmap(np.linspace(0.1, 0.9, len(key_layers)))

    for ci, (name, idx) in enumerate(key_layers.items()):
        il = I_latent[idx, mask_ep]
        ii = I_ind[idx, mask_ep]
        ratio = np.where((ii > 0) & ~np.isnan(il), il / ii, np.nan)
        valid = ~np.isnan(ratio)
        if valid.any():
            ax.plot(xdata[valid], ratio[valid], 'o-', c=clrs[ci], lw=2.5, ms=6,
                    label=name, alpha=0.85)

    ax.axhline(1.0, ls='--', c='grey', alpha=0.5, label='independent')
    ax.set_xlabel(r'$\log_{10}(\mathrm{epoch}+1)$')
    ax.set_ylabel('I(Z;Y) / I_ind')
    ax.set_title('Redundancy–synergy dynamics')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'redundancy_synergy_mlp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR / "redundancy_synergy_mlp.png"}')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    labels_train = load_labels('train')
    labels_test = load_labels('test')
    print(f'Epochs: {EPOCHS}')
    print(f'Layers: {len(LAYERS)}')
    print(f'Train: {labels_train.shape}, Test: {labels_test.shape}')

    npz_path = OUT_DIR / 'latent_mi_mlp.npz'
    if npz_path.exists():
        print('Loading precomputed results...')
        data = np.load(npz_path)
        I_latent = data['I_latent']
        I_ind = data['I_ind']
        H_gauss_arr = data['H_gauss']
        acc_z_arr = data['acc_z']
        acc_full_arr = data['acc_full']
        mse_arr = data['mse']
    else:
        print('\n=== Computing latent MI (MLP head) ===')
        I_latent, I_ind, H_gauss_arr, acc_z_arr, acc_full_arr, mse_arr = compute_all(
            labels_train, labels_test)
        np.savez(npz_path,
                 I_latent=I_latent, I_ind=I_ind, H_gauss=H_gauss_arr,
                 acc_z=acc_z_arr, acc_full=acc_full_arr, mse=mse_arr,
                 layers=LAYERS, epochs=EPOCHS)
        print(f'Saved: {npz_path}')

    print('\n=== Plotting ===')
    plot_info_plane(I_latent, H_gauss_arr)
    plot_by_layer(I_latent, I_ind, acc_z_arr, acc_full_arr)
    plot_redundancy_synergy_dynamics(I_latent, I_ind)

    # Summary with Fano bounds
    H_Y = np.log2(N_CLASSES)
    SHORT = short_names()

    print('\n=== Summary (epoch 900) ===')
    print(f'H(Y) = {H_Y:.3f} bits')
    print(f'{"Layer":16s} {"I(Z;Y)":>8s} {"Fano_lb":>8s} {"Fano_ub":>8s} '
          f'{"acc_z":>6s} {"acc_T":>6s} {"ratio":>6s} {"R/S":>8s}')
    print('-' * 80)

    for i in range(len(LAYERS)):
        il = I_latent[i, -1]
        az = acc_z_arr[i, -1]
        af = acc_full_arr[i, -1]
        ii = I_ind[i, -1]

        fb = fano_bound(az)
        gap = fano_gap_upper(az)
        upper = min(H_Y, il + gap)
        ratio = az / af if af > 0 and not np.isnan(az) else np.nan
        rs = il / ii if ii > 0 and not np.isnan(il) else np.nan

        print(f'{SHORT[i]:16s} {il:8.3f} {fb:8.3f} {upper:8.3f} '
              f'{az:6.3f} {af:6.3f} {ratio:6.3f} {rs:8.4f}')

    print('\nDone!')
