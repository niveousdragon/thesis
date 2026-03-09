"""
08_mei_vs_ib.py — MEI data analysis in the context of Information Bottleneck findings.

7-panel figure: selectivity timing, fraction selective vs I(Z;Y), MEI entropy vs I(Z;Y),
complexity vs ID, labile neurons, class distribution heatmap, activation heatmap.
"""

import sys
sys.path.insert(0, r'C:\Users\User\PycharmProjects\driada\src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from driada.utils.plot import make_beautiful

# ── Configuration ──────────────────────────────────────────────────────────
OUT_PATH = "results/mei_vs_ib.png"

MEI_LAYERS = [
    'sn1', 'layer1.0.sn1', 'layer1.1.sn1', 'layer2.0.sn1', 'layer2.1.sn1',
    'layer3.0.sn1', 'layer3.1.sn1', 'layer4.0.sn1', 'layer4.1.sn1'
]
# Mapping MEI layers → activity layer indices (17 layers, sn1 only)
MEI_TO_ACT = {
    'sn1': 0, 'layer1.0.sn1': 1, 'layer1.1.sn1': 3,
    'layer2.0.sn1': 5, 'layer2.1.sn1': 7,
    'layer3.0.sn1': 9, 'layer3.1.sn1': 11,
    'layer4.0.sn1': 13, 'layer4.1.sn1': 15
}

ACT_EPOCHS = [0,5,10,20,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300,400,500,600,700,800,900]

# Short labels for layers
SHORT_LABELS = {
    'sn1': 'sn1', 'layer1.0.sn1': '1.0', 'layer1.1.sn1': '1.1',
    'layer2.0.sn1': '2.0', 'layer2.1.sn1': '2.1',
    'layer3.0.sn1': '3.0', 'layer3.1.sn1': '3.1',
    'layer4.0.sn1': '4.0', 'layer4.1.sn1': '4.1'
}

KEY_LAYERS = ['sn1', 'layer2.1.sn1', 'layer3.1.sn1', 'layer4.1.sn1']
KEY_COLORS = {'sn1': '#1f77b4', 'layer2.1.sn1': '#ff7f0e',
              'layer3.1.sn1': '#2ca02c', 'layer4.1.sn1': '#d62728'}

STYLE_KW = dict(spine_width=2, tick_width=2, tick_length=5, tick_pad=6,
                tick_labelsize=11, label_size=13, title_size=14,
                legend_fontsize=9, lowercase_labels=False)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("results/mei_summary.csv")
mi_data = np.load("results/latent_mi.npz")
dim_data = np.load("results/dimensionality.npz")

I_latent = mi_data['I_latent']      # (17, 26)
id_nn = dim_data['id_nn']           # (17, 26)

MEI_EPOCHS = sorted(df['epoch'].unique())

# Find common epochs between MEI and activity
common_epochs = sorted(set(MEI_EPOCHS) & set(ACT_EPOCHS))
mei_epoch_idx = {e: MEI_EPOCHS.index(e) for e in common_epochs if e in MEI_EPOCHS}
act_epoch_idx = {e: ACT_EPOCHS.index(e) for e in common_epochs}

print(f"MEI epochs: {MEI_EPOCHS}")
print(f"Common epochs: {common_epochs}")
print(f"MEI layers: {MEI_LAYERS}")
print(f"Neurons per layer: {df.groupby('layer')['neuron'].nunique().values[0]}")


# ── Helper: selectivity criterion ──────────────────────────────────────────
def compute_selectivity(df_sub):
    """For a group (layer, epoch, neuron), check if all 3 methods agree on class
    with max_prob >= 0.75 and max_activation >= 0.75."""
    if len(df_sub) < 3:
        return False, -1
    classes = df_sub['argmax_class'].values
    if not (classes[0] == classes[1] == classes[2]):
        return False, -1
    if (df_sub['max_prob'].values >= 0.75).all() and (df_sub['max_activation'].values >= 0.75).all():
        return True, int(classes[0])
    return False, -1


def get_consensus_class(df_sub):
    """Return consensus argmax_class if all 3 methods agree, else -1."""
    if len(df_sub) < 3:
        return -1
    classes = df_sub['argmax_class'].values
    if classes[0] == classes[1] == classes[2]:
        return int(classes[0])
    return -1


# ── Precompute selectivity per (layer, epoch, neuron) ─────────────────────
print("Computing selectivity...")
sel_records = []
for (layer, epoch, neuron), grp in df.groupby(['layer', 'epoch', 'neuron']):
    is_sel, cls = compute_selectivity(grp)
    cons_cls = get_consensus_class(grp)
    sel_records.append({
        'layer': layer, 'epoch': epoch, 'neuron': neuron,
        'selective': is_sel, 'sel_class': cls, 'consensus_class': cons_cls
    })
sel_df = pd.DataFrame(sel_records)

# ── FIGURE SETUP ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14))
gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.45)

# ── Panel 1: First epoch of selectivity vs ID minimum ─────────────────────
ax1 = fig.add_subplot(gs[0, 0])

first_sel_epoch = {}
for layer in MEI_LAYERS:
    epochs_first = []
    for neuron in range(64):
        sub = sel_df[(sel_df['layer'] == layer) & (sel_df['neuron'] == neuron) & sel_df['selective']]
        if len(sub) > 0:
            epochs_first.append(sub['epoch'].min())
    first_sel_epoch[layer] = np.median(epochs_first) if epochs_first else np.nan

# ID minimum epoch per layer (skip epoch 0, ignore NaN)
id_min_epoch = {}
for layer in MEI_LAYERS:
    act_idx = MEI_TO_ACT[layer]
    id_curve = id_nn[act_idx, :]
    # Search from epoch 5 onwards (index 1), mask NaN
    search_start = 1  # skip epoch 0
    valid_curve = id_curve[search_start:]
    valid_epochs = np.array(ACT_EPOCHS[search_start:])
    mask = ~np.isnan(valid_curve)
    if mask.any():
        min_idx = np.nanargmin(valid_curve)
        id_min_epoch[layer] = int(valid_epochs[min_idx])
    else:
        id_min_epoch[layer] = np.nan

# Only layers with real V-shape in ВР
VSHAPE_LAYERS = ['layer3.1.sn1', 'layer4.0.sn1', 'layer4.1.sn1']

x_pos = np.arange(len(MEI_LAYERS))
labels = [SHORT_LABELS[l] for l in MEI_LAYERS]

# Panel A: bar chart for V-shape layers only
x_pos_v = np.arange(len(VSHAPE_LAYERS))
labels_v = [SHORT_LABELS[l] for l in VSHAPE_LAYERS]

ax1.bar(x_pos_v - 0.17, [id_min_epoch[l] for l in VSHAPE_LAYERS], 0.3,
        color='#1f77b4', label='ID minimum epoch', edgecolor='black', linewidth=0.8)
ax1.bar(x_pos_v + 0.17, [first_sel_epoch[l] for l in VSHAPE_LAYERS], 0.3,
        color='#d62728', label='Median first selectivity', edgecolor='black', linewidth=0.8)
for i, l in enumerate(VSHAPE_LAYERS):
    ax1.text(i - 0.17, id_min_epoch[l] + 3, f'{id_min_epoch[l]:.0f}',
             ha='center', va='bottom', fontsize=10, color='#1f77b4', fontweight='bold')
    ax1.text(i + 0.17, first_sel_epoch[l] + 3, f'{first_sel_epoch[l]:.0f}',
             ha='center', va='bottom', fontsize=10, color='#d62728', fontweight='bold')
ax1.set_xticks(x_pos_v)
ax1.set_xticklabels(labels_v)
ax1.set_xlabel("Layer")
ax1.set_ylabel("Epoch")
ax1.set_title("A. Compression precedes selectivity\n(layers with V-shape ID only)")
ax1.legend(fontsize=8)
make_beautiful(ax1, **STYLE_KW)

# ── Panel 2: Fraction selective vs I(Z;Y) for key layers ──────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2b = ax2.twinx()

for layer in KEY_LAYERS:
    frac_sel = []
    izy_vals = []
    epochs_plot = []
    act_idx = MEI_TO_ACT[layer]
    for e in common_epochs:
        sub = sel_df[(sel_df['layer'] == layer) & (sel_df['epoch'] == e)]
        if len(sub) > 0:
            frac_sel.append(sub['selective'].mean())
            izy_vals.append(I_latent[act_idx, act_epoch_idx[e]])
            epochs_plot.append(e)
    c = KEY_COLORS[layer]
    sl = SHORT_LABELS[layer]
    ax2.plot(epochs_plot, frac_sel, '-', color=c, linewidth=2, label=f'{sl} frac')
    ax2b.plot(epochs_plot, izy_vals, '--', color=c, linewidth=1.5, label=f'{sl} I(Z;Y)')

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Fraction selective")
ax2b.set_ylabel("I(Z;Y)")
ax2.set_title("B. Selectivity fraction vs I(Z;Y)")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, ncol=2, loc='upper left')
make_beautiful(ax2, **STYLE_KW)
# Style the twin axis too (only right spine)
for sp in ['top', 'left']:
    ax2b.spines[sp].set_visible(False)
ax2b.spines['right'].set_linewidth(STYLE_KW['spine_width'])
ax2b.tick_params(axis='y', labelsize=STYLE_KW['tick_labelsize'],
                 width=STYLE_KW['tick_width'], length=STYLE_KW['tick_length'])

# ── Panel 3: MEI entropy vs I(Z;Y) scatter ────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])

layer_groups = {
    'early (sn1–1.1)': ['sn1', 'layer1.0.sn1', 'layer1.1.sn1'],
    'mid (2.0–2.1)': ['layer2.0.sn1', 'layer2.1.sn1'],
    'late-mid (3.0–3.1)': ['layer3.0.sn1', 'layer3.1.sn1'],
    'deep (4.0–4.1)': ['layer4.0.sn1', 'layer4.1.sn1']
}
group_colors = {'early (sn1–1.1)': '#1f77b4', 'mid (2.0–2.1)': '#ff7f0e',
                'late-mid (3.0–3.1)': '#2ca02c', 'deep (4.0–4.1)': '#d62728'}

for gname, glayers in layer_groups.items():
    xs, ys = [], []
    for layer in glayers:
        act_idx = MEI_TO_ACT[layer]
        for e in common_epochs:
            sub = df[(df['layer'] == layer) & (df['epoch'] == e)]
            if len(sub) > 0:
                xs.append(sub['rel_perpl'].mean())
                ys.append(I_latent[act_idx, act_epoch_idx[e]])
    ax3.scatter(xs, ys, c=group_colors[gname], label=gname, alpha=0.6, s=25, edgecolors='none')

ax3.set_xlabel("Mean rel. perplexity")
ax3.set_ylabel("I(Z;Y)")
ax3.set_title("C. MEI entropy vs I(Z;Y)")
ax3.legend(fontsize=7, loc='best')
make_beautiful(ax3, **STYLE_KW)

# ── Panel 4: Complexity vs ID for key layers ───────────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
ax4b = ax4.twinx()

for layer in KEY_LAYERS:
    compl_vals = []
    id_vals = []
    epochs_plot = []
    act_idx = MEI_TO_ACT[layer]
    for e in common_epochs:
        sub = df[(df['layer'] == layer) & (df['epoch'] == e)]
        if len(sub) > 0:
            compl_vals.append(sub['complexity'].mean())
            id_vals.append(id_nn[act_idx, act_epoch_idx[e]])
            epochs_plot.append(e)
    c = KEY_COLORS[layer]
    sl = SHORT_LABELS[layer]
    ax4.plot(epochs_plot, compl_vals, '-', color=c, linewidth=2, label=f'{sl} compl')
    ax4b.plot(epochs_plot, id_vals, '--', color=c, linewidth=1.5, label=f'{sl} ID')

ax4.set_xlabel("Epoch")
ax4.set_ylabel("Mean complexity")
ax4b.set_ylabel("Intrinsic dim (ID)")
ax4.set_title("D. MEI complexity vs ID")
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=7, ncol=2, loc='upper left')
make_beautiful(ax4, **STYLE_KW)
for sp in ['top', 'left']:
    ax4b.spines[sp].set_visible(False)
ax4b.spines['right'].set_linewidth(STYLE_KW['spine_width'])
ax4b.tick_params(axis='y', labelsize=STYLE_KW['tick_labelsize'],
                 width=STYLE_KW['tick_width'], length=STYLE_KW['tick_length'])

# ── Panel 5: Labile neurons by layer (among selective only) ───────────────
ax5 = fig.add_subplot(gs[1, 0])

# Labile = selective neuron that changed its selective class at least once
labile_frac = {}
for layer in MEI_LAYERS:
    sorted_epochs = sorted(sel_df[sel_df['layer'] == layer]['epoch'].unique())
    n_labile = 0
    n_sel_multiple = 0
    for neuron in range(64):
        sel_classes = []
        for e in sorted_epochs:
            row = sel_df[(sel_df['layer'] == layer) & (sel_df['epoch'] == e) & (sel_df['neuron'] == neuron)]
            if len(row) > 0 and row['selective'].values[0]:
                sel_classes.append(row['sel_class'].values[0])
        if len(sel_classes) >= 2:
            n_sel_multiple += 1
            if len(set(sel_classes)) > 1:
                n_labile += 1
    labile_frac[layer] = n_labile / n_sel_multiple if n_sel_multiple > 0 else 0

bars = ax5.bar(x_pos, [labile_frac[l] for l in MEI_LAYERS], color='#9467bd', edgecolor='black', linewidth=0.8)
# Add counts on top
for i, layer in enumerate(MEI_LAYERS):
    sorted_epochs = sorted(sel_df[sel_df['layer'] == layer]['epoch'].unique())
    n_sel = sum(1 for n in range(64)
                if sum(1 for e in sorted_epochs
                       if len(sel_df[(sel_df['layer']==layer)&(sel_df['epoch']==e)&(sel_df['neuron']==n)]) > 0
                       and sel_df[(sel_df['layer']==layer)&(sel_df['epoch']==e)&(sel_df['neuron']==n)]['selective'].values[0]
                ) >= 2)
    val = labile_frac[layer]
    if val > 0 or n_sel > 0:
        ax5.text(i, val + 0.01, f'{val:.0%}', ha='center', va='bottom', fontsize=8)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(labels, rotation=45)
ax5.set_xlabel("Layer")
ax5.set_ylabel("Fraction labile (among selective)")
ax5.set_title("E. Labile selective neurons by layer")
make_beautiful(ax5, **STYLE_KW)

# ── Panel 6: Class distribution heatmap (epoch 700) ───────────────────────
ax6 = fig.add_subplot(gs[1, 1])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
heatmap_data = np.zeros((len(MEI_LAYERS), 10))
for i, layer in enumerate(MEI_LAYERS):
    sub = sel_df[(sel_df['layer'] == layer) & (sel_df['epoch'] == 700)]
    for _, row in sub.iterrows():
        cc = row['consensus_class']
        if cc >= 0:
            heatmap_data[i, cc] += 1

im = ax6.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax6.set_yticks(range(len(MEI_LAYERS)))
ax6.set_yticklabels(labels)
ax6.set_xticks(range(10))
ax6.set_xticklabels([c[:4] for c in class_names], rotation=60, fontsize=8)
ax6.set_xlabel("Class")
ax6.set_ylabel("Layer")
ax6.set_title("F. Class distribution (epoch 700)")
plt.colorbar(im, ax=ax6, shrink=0.8, label='# neurons')
# Don't call make_beautiful on heatmap — it hides spines we want
ax6.tick_params(labelsize=STYLE_KW['tick_labelsize'])

# ── Panel 7: Max activation heatmap ───────────────────────────────────────
ax7 = fig.add_subplot(gs[1, 2:])

act_heatmap = np.zeros((len(MEI_LAYERS), len(MEI_EPOCHS)))
for i, layer in enumerate(MEI_LAYERS):
    for j, epoch in enumerate(MEI_EPOCHS):
        sub = df[(df['layer'] == layer) & (df['epoch'] == epoch)]
        act_heatmap[i, j] = sub['max_activation'].mean() if len(sub) > 0 else np.nan

im7 = ax7.imshow(act_heatmap, aspect='auto', cmap='viridis', interpolation='nearest')
ax7.set_yticks(range(len(MEI_LAYERS)))
ax7.set_yticklabels(labels)
ax7.set_xticks(range(len(MEI_EPOCHS)))
ax7.set_xticklabels(MEI_EPOCHS, rotation=60, fontsize=8)
ax7.set_xlabel("Epoch")
ax7.set_ylabel("Layer")
ax7.set_title("G. Mean max activation by layer × epoch")
plt.colorbar(im7, ax=ax7, shrink=0.8, label='Mean max activation')

# Highlight layer3.1 row
layer31_idx = MEI_LAYERS.index('layer3.1.sn1')
ax7.axhline(y=layer31_idx - 0.5, color='red', linewidth=2, linestyle='--')
ax7.axhline(y=layer31_idx + 0.5, color='red', linewidth=2, linestyle='--')
ax7.text(len(MEI_EPOCHS) + 0.3, layer31_idx, '← 3.1', color='red', fontsize=12,
         va='center', fontweight='bold', clip_on=False)

ax7.tick_params(labelsize=STYLE_KW['tick_labelsize'])

# ── Save ───────────────────────────────────────────────────────────────────
fig.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nFigure saved to {OUT_PATH}")

# ── Summary statistics ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print("\n--- Panel A: First selectivity epoch & ID minimum ---")
for l in MEI_LAYERS:
    sl = SHORT_LABELS[l]
    fse = first_sel_epoch[l]
    idm = id_min_epoch[l]
    print(f"  {sl:>5s}: first_sel_epoch = {fse:>6.0f},  ID_min_epoch = {idm:>4d}")

print("\n--- Panel B: Fraction selective at epoch 700 ---")
for l in KEY_LAYERS:
    sub = sel_df[(sel_df['layer'] == l) & (sel_df['epoch'] == 700)]
    fs = sub['selective'].mean() if len(sub) > 0 else 0
    print(f"  {SHORT_LABELS[l]:>5s}: {fs:.3f}")

print("\n--- Panel C: Correlation (mean rel_perpl, I(Z;Y)) ---")
all_rp, all_izy = [], []
for layer in MEI_LAYERS:
    act_idx = MEI_TO_ACT[layer]
    for e in common_epochs:
        sub = df[(df['layer'] == layer) & (df['epoch'] == e)]
        if len(sub) > 0:
            all_rp.append(sub['rel_perpl'].mean())
            all_izy.append(I_latent[act_idx, act_epoch_idx[e]])
all_rp, all_izy = np.array(all_rp), np.array(all_izy)
valid = np.isfinite(all_rp) & np.isfinite(all_izy)
corr = np.corrcoef(all_rp[valid], all_izy[valid])[0, 1]
print(f"  Pearson r = {corr:.4f} (N={valid.sum()} valid / {len(all_rp)} total points)")

print("\n--- Panel E: Labile fraction by layer (among selective) ---")
for l in MEI_LAYERS:
    print(f"  {SHORT_LABELS[l]:>5s}: {labile_frac[l]:.3f}")

print("\n--- Panel F: Class distribution entropy at epoch 700 ---")
for i, l in enumerate(MEI_LAYERS):
    row = heatmap_data[i]
    total = row.sum()
    if total > 0:
        p = row[row > 0] / total
        ent = -np.sum(p * np.log2(p))
    else:
        ent = 0
    print(f"  {SHORT_LABELS[l]:>5s}: {total:>2.0f} neurons with consensus, entropy = {ent:.2f} bits")

print("\n--- Panel G: Layer3.1 activation anomaly ---")
l31_acts = act_heatmap[layer31_idx, :]
mean_other = np.nanmean(np.delete(act_heatmap, layer31_idx, axis=0), axis=0)
print(f"  Layer 3.1 mean activation across epochs: {np.nanmean(l31_acts):.4f}")
print(f"  Other layers mean activation across epochs: {np.nanmean(mean_other):.4f}")
print(f"  Layer 3.1 min activation: {np.nanmin(l31_acts):.4f} at epoch {MEI_EPOCHS[np.nanargmin(l31_acts)]}")
print(f"  Layer 3.1 max activation: {np.nanmax(l31_acts):.4f} at epoch {MEI_EPOCHS[np.nanargmax(l31_acts)]}")

print("\n" + "="*70)
print("Done.")
