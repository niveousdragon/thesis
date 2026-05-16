"""Convert MANGO evolution YAML to CSV + NPZ for use without jax.

Run in mango conda env: conda run -n mango python _convert_mei_yaml.py

Outputs:
  - mango_data/results/mei_summary.csv  (max_a, complexity, rel_perpl, argmax_class, max_prob)
  - mango_data/results/mei_probs.npz    (full 10-class probability distributions)
"""
import yaml
import numpy as np
import csv
from pathlib import Path

YAML_PATH = Path('C:/Users/User/PycharmProjects/thesis/mango_data/SJ-SNN-T50/SJ-SNN evolution data full 3.yaml')
OUT_CSV = Path('C:/Users/User/PycharmProjects/thesis/mango_data/results/mei_summary.csv')
OUT_NPZ = Path('C:/Users/User/PycharmProjects/thesis/mango_data/results/mei_probs.npz')

LABEL_NAMES = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck',
}

print(f'Loading {YAML_PATH}...')
with open(YAML_PATH, 'r') as f:
    data = yaml.unsafe_load(f)

epochs = sorted(data.keys())
print(f'Epochs: {epochs}')

rows = []
all_probs = []  # list of (10,) arrays

for epoch in epochs:
    layers = data[epoch]
    for layer in sorted(layers.keys()):
        neurons = layers[layer]
        for neuron_id, methods in neurons.items():
            neuron_idx = int(np.array(neuron_id))
            for method_key, info in methods.items():
                max_a = float(np.array(info.get('max_a', np.nan)))
                complexity = info.get('complexity', None)
                rel_perpl = float(info.get('rel_perpl', np.nan))

                probs = info.get('probs', None)
                if probs is not None:
                    probs = np.array(probs, dtype=np.float32)
                    argmax_class = int(np.argmax(probs))
                    max_prob = float(np.max(probs))
                    all_probs.append(probs)
                else:
                    argmax_class = -1
                    max_prob = np.nan
                    all_probs.append(np.full(10, np.nan, dtype=np.float32))

                rows.append({
                    'epoch': epoch,
                    'layer': layer,
                    'neuron': neuron_idx,
                    'method': method_key,
                    'max_activation': max_a,
                    'complexity': complexity,
                    'rel_perpl': rel_perpl,
                    'argmax_class': argmax_class,
                    'class_name': LABEL_NAMES.get(argmax_class, ''),
                    'max_prob': max_prob,
                })

print(f'Total rows: {len(rows)}')
print(f'Sample: {rows[0]}')

# CSV
OUT_CSV.parent.mkdir(exist_ok=True)
fieldnames = ['epoch', 'layer', 'neuron', 'method', 'max_activation',
              'complexity', 'rel_perpl', 'argmax_class', 'class_name', 'max_prob']
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f'Saved: {OUT_CSV}')

# NPZ with full probability distributions
probs_array = np.stack(all_probs)  # (n_rows, 10)
np.savez_compressed(OUT_NPZ, probs=probs_array,
                    epochs=np.array(epochs),
                    rows_info='Same order as mei_summary.csv')
print(f'Saved: {OUT_NPZ}, shape={probs_array.shape}')
