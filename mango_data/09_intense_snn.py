"""
09: INTENSE на активности SNN.

Применяем INTENSE (из DRIADA) к данным активности SNN × метки классов.
Цель: проверить согласованность с exact discrete MI (mutual_info_score),
получить p-values для каждого нейрона.

Для каждого (epoch, layer):
  - ts_bunch1 = [TimeSeries(act[:, i]) for i in range(N)]  — нейроны
  - ts_bunch2 = [TimeSeries(labels)]  — метки классов
  - compute_me_stats → MI + p-values

API compute_me_stats возвращает:
  stats[name1][name2] = {'me': float, 'pval': float, 'rval': float, ...}
  significance[name1][name2] = {'stage1': bool, 'stage2': bool}

Нейрон значим ↔ significance[name]['class']['stage2'] == True
MI хранится в 'me' (не 'mi'!), единицы — nats (sklearn mutual_info_score тоже в nats).
"""

import sys
import gc
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')

from sklearn.metrics import mutual_info_score
from driada.information.info_base import TimeSeries
from driada.intense.intense_base import compute_me_stats

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

# Подмножество эпох для анализа
# Ключевые: начало, компрессия, после компрессии, финал
KEY_EPOCHS = [0, 10, 20, 30, 50, 100, 200, 500, 700, 900]

# Подмножество слоёв: по одному из каждой группы + sn1
KEY_LAYERS = [
    'sn1',
    'layer1.1.sn2',
    'layer2.1.sn2',
    'layer3.1.sn2',
    'layer4.1.sn2',
]


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
    return arr.sum(axis=0)  # (n_samples, N)


def run_intense_for_layer(act, labels, layer_name, epoch, verbose=True):
    """
    Запускает INTENSE для одного слоя/эпохи.
    Returns: list of dicts с результатами для каждого нейрона.
    """
    N = act.shape[1]
    neuron_names = [f'n{i}' for i in range(N)]

    # TimeSeries для каждого нейрона (spike counts → auto-detected as discrete)
    ts_neurons = [TimeSeries(act[:, i].astype(float), name=nm)
                  for i, nm in enumerate(neuron_names)]

    # Метки классов — одна "фича"
    ts_labels = [TimeSeries(labels.astype(float), name='class')]

    # Запуск INTENSE
    stats, significance, info = compute_me_stats(
        ts_bunch1=ts_neurons,
        ts_bunch2=ts_labels,
        names1=neuron_names,
        names2=['class'],
        mode='two_stage',
        metric='mi',
        n_shuffles_stage1=100,
        n_shuffles_stage2=10000,
        find_optimal_delays=False,
        pval_thr=0.01,
        multicomp_correction='holm',
        verbose=verbose,
        seed=42,
        enable_parallelization=True,
        n_jobs=-1,
    )

    # Извлечь результаты
    results = []
    for i in range(N):
        nm = neuron_names[i]
        s = stats.get(nm, {}).get('class', {})
        sig = significance.get(nm, {}).get('class', {})

        mi_intense = s.get('me', np.nan)      # MI из INTENSE (nats)
        pval = s.get('pval', np.nan)           # p-value (stage2, parametric)
        rval = s.get('rval', s.get('pre_rval', np.nan))  # rank
        is_sig = sig.get('stage2', False)      # прошёл оба этапа

        # Exact discrete MI для сравнения (nats)
        mi_exact = mutual_info_score(labels, act[:, i].astype(int))

        results.append({
            'layer': layer_name,
            'epoch': epoch,
            'neuron': i,
            'mi_intense': mi_intense,
            'mi_exact': mi_exact,
            'pval': pval,
            'rval': rval,
            'significant': is_sig,
        })

    return results


if __name__ == '__main__':
    labels = load_labels('test')

    # Фильтруем эпохи по доступным
    epochs_to_run = [e for e in KEY_EPOCHS if e in EPOCHS]
    layers_to_run = KEY_LAYERS

    print(f'INTENSE на SNN активности')
    print(f'Эпохи: {epochs_to_run}')
    print(f'Слои: {layers_to_run}')
    print(f'Всего комбинаций: {len(epochs_to_run) * len(layers_to_run)}')

    all_results = []

    for epoch in epochs_to_run:
        for layer in layers_to_run:
            print(f'\n=== Epoch {epoch}, {layer} ===')
            act = load_activity(epoch, layer, split='test')
            if act is None:
                print('  Нет данных, пропуск')
                continue

            N = act.shape[1]
            print(f'  N={N}, samples={act.shape[0]}')

            try:
                results = run_intense_for_layer(act, labels, layer, epoch)
                all_results.extend(results)

                # Краткая сводка
                df = pd.DataFrame(results)
                n_sig = df['significant'].sum()
                mi_int = df['mi_intense'].dropna()
                mi_ext = df['mi_exact']
                if len(mi_int) > 1 and mi_int.std() > 0:
                    corr = mi_int.corr(mi_ext.loc[mi_int.index])
                else:
                    corr = np.nan
                print(f'  Selective: {n_sig}/{N} ({100*n_sig/N:.0f}%)')
                print(f'  Corr(INTENSE MI, exact MI): {corr:.4f}')
            except Exception as e:
                import traceback
                print(f'  Ошибка: {e}')
                traceback.print_exc()

            gc.collect()

    # Сохранение
    df_all = pd.DataFrame(all_results)
    csv_path = OUT_DIR / 'intense_snn.csv'
    df_all.to_csv(csv_path, index=False)
    print(f'\nСохранено: {csv_path}')
    print(f'Всего записей: {len(df_all)}')

    # Итоговая сводка
    print('\n=== Итоги ===')
    print(f'{"Layer":20s} {"Epoch":>6s} {"N_sig":>6s} {"N_tot":>6s} '
          f'{"frac":>6s} {"r(MI)":>8s}')
    print('-' * 60)
    for (layer, epoch), grp in df_all.groupby(['layer', 'epoch']):
        n_sig = grp['significant'].sum()
        n_tot = len(grp)
        mi_int = grp['mi_intense'].dropna()
        mi_ext = grp['mi_exact'].loc[mi_int.index]
        if len(mi_int) > 1 and mi_int.std() > 0:
            corr = mi_int.corr(mi_ext)
        else:
            corr = np.nan
        print(f'{layer:20s} {epoch:6d} {n_sig:6d} {n_tot:6d} '
              f'{n_sig/n_tot:6.2f} {corr:8.4f}')
