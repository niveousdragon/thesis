"""Inspect MEI pickle file structure."""
import pickle
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else (
    'C:/Users/User/PycharmProjects/thesis/mango_data/SJ-SNN-T50/'
    'SJ-SNN iter 200/sjsnn-cifar10-gan_sn-layer4.0.sn1_processed/'
    'dat/opt_info_u0_layer4.0.sn1.pkl'
)

with open(path, 'rb') as f:
    data = pickle.load(f)

print(f'Type: {type(data).__name__}, keys: {len(data)}')
print(f'Keys: {list(data.keys())[:10]}...')

# Inspect first entry in detail
first_key = list(data.keys())[0]
val = data[first_key]
print(f'\nFirst key: {first_key}')
print(f'  Type: {type(val).__name__}, len={len(val)}')
for i, item in enumerate(val):
    itype = type(item).__name__
    if hasattr(item, 'shape'):
        print(f'  [{i}]: {itype} shape={item.shape} dtype={item.dtype}')
        if item.size < 20:
            print(f'       values: {np.array(item)}')
    elif isinstance(item, list):
        print(f'  [{i}]: list len={len(item)}, first few: {item[:5]}')
    else:
        print(f'  [{i}]: {itype} = {repr(item)[:200]}')
