"""
07b: Контроль latent_dim=8 для I(Z;Y).
Проверяет, сохраняется ли hunchback I(Z;Y) при другом размере латентного пространства.
Только I(Z;Y) и accuracy — I_ind и H_gauss не зависят от latent_dim.
"""

import sys
import gc
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path

sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')

from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from driada.dim_reduction.flexible_ae import ModularAutoencoder
from driada.information.ksg import nonparam_mi_cd

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
N_CLASSES = 10
TRAIN_SUBSAMPLE = 10000

# Changed parameter
LATENT_DIM = 8
HIDDEN_DIM = 256
AE_EPOCHS = 50
AE_LR = 1e-3
AE_BATCH = 128
CLASS_WEIGHT = 1.0


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
    arr = data[key]
    return arr.sum(axis=0)


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


def train_ae_classifier(act_train, labels_train, act_test, labels_test):
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
        cls_loss = model.losses[1]
        logits = cls_loss.classifier(model.encode(test_tensor))
        preds = logits.argmax(dim=1).cpu().numpy()
        acc_z = np.mean(preds == labels_test)

    return Z_test, acc_z


def baseline_accuracy(act_train, labels_train, act_test, labels_test):
    lr = LogisticRegression(max_iter=500, solver='lbfgs', n_jobs=-1)
    lr.fit(act_train, labels_train)
    return lr.score(act_test, labels_test)


if __name__ == '__main__':
    labels_train = load_labels('train')
    labels_test = load_labels('test')

    npz_path = OUT_DIR / 'latent_mi_dim8.npz'
    print(f'LATENT_DIM = {LATENT_DIM}')
    print(f'Epochs: {len(EPOCHS)}, Layers: {len(LAYERS)}')

    n_layers = len(LAYERS)
    n_epochs = len(EPOCHS)

    I_latent = np.full((n_layers, n_epochs), np.nan)
    acc_z = np.full((n_layers, n_epochs), np.nan)

    print('\n=== Computing latent MI (dim=8) ===')
    for j, epoch in enumerate(EPOCHS):
        print(f'  Epoch {epoch}...', end=' ', flush=True)
        for i, layer in enumerate(LAYERS):
            act_test = load_activity(epoch, layer, split='test')
            if act_test is None:
                continue

            # Load train — use test-sized subsample to avoid OOM on layer4 train
            try:
                act_train_full = load_activity(epoch, layer, split='train')
            except Exception:
                act_train_full = None
            if act_train_full is None:
                # Fallback: split test 50/50
                n = act_test.shape[0]
                idx = np.random.RandomState(42).permutation(n)
                act_train = act_test[idx[:n//2]]
                lab_train = labels_test[idx[:n//2]]
                act_test_use = act_test[idx[n//2:]]
                lab_test_use = labels_test[idx[n//2:]]
            else:
                act_train, lab_train = stratified_subsample(act_train_full, labels_train)
                del act_train_full
                act_test_use = act_test
                lab_test_use = labels_test

            try:
                Z_test, az = train_ae_classifier(act_train, lab_train, act_test_use, lab_test_use)
                acc_z[i, j] = az
                I_latent[i, j] = nonparam_mi_cd(Z_test, lab_test_use, k=5, base=2.0)
            except Exception as e:
                print(f'err {layer}: {e}', end=' ')

            del act_test, act_train
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print('done')

    np.savez(npz_path,
             I_latent=I_latent, acc_z=acc_z,
             layers=LAYERS, epochs=EPOCHS)
    print(f'\nSaved: {npz_path}')

    # Summary for final epoch
    SHORT = []
    for l in LAYERS:
        if l == 'sn1':
            SHORT.append('sn1')
        else:
            parts = l.split('.')
            block = parts[0].replace('layer', 'L')
            SHORT.append(f'{block}.{parts[1]}.{parts[2][-1]}')

    print(f'\n{"Layer":16s} {"I(Z;Y) d=8":>12s} {"I(Z;Y) d=16":>12s} {"acc_Z d=8":>10s}')
    print('-' * 50)

    # Load dim=16 for comparison
    d16 = np.load(OUT_DIR / 'latent_mi.npz')
    I_16 = d16['I_latent']

    for i in range(n_layers):
        il8 = I_latent[i, -1]
        il16 = I_16[i, -1]
        az = acc_z[i, -1]
        print(f'{SHORT[i]:16s} {il8:12.3f} {il16:12.3f} {az:10.3f}')

    print('\nDone!')
