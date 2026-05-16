#!/usr/bin/env python
"""
Estimate effective embedding dimension for calcium signals via PCA on Takens embedding.

Idea: embed with large m, do PCA on embedded data, find how many components
carry signal above the noise floor. This gives the "signal dimension" —
not the attractor dimension (FNN), but the dimension of the informative
part of the embedding.

Methods tested:
  1. Broken stick: compare explained variance to broken-stick null model
  2. Noise threshold: components with eigenvalue > noise eigenvalue estimate
  3. Participation ratio: effective number of dimensions

Tested on synthetic HD neurons where we know the answer should be small (1-2).

Usage:
    python estimate_signal_dim.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DRIADA_ROOT = Path(r'C:\Users\User\PycharmProjects\driada')
sys.path.insert(0, str(DRIADA_ROOT / 'src'))

from driada.experiment.synthetic import generate_tuned_selectivity_exp
from driada.recurrence.embedding import estimate_tau, takens_embedding

OUT = Path(__file__).parent / 'results' / 'synthetic_hd'
OUT.mkdir(parents=True, exist_ok=True)


def broken_stick(n):
    """Broken stick expected values for n components."""
    bs = np.zeros(n)
    for i in range(n):
        bs[i] = sum(1.0 / (j + 1) for j in range(i, n))
    return bs / n


def signal_dim_pca(data_1d, tau, max_m=15):
    """Estimate signal dimension via PCA on Takens embedding.

    For each m from 2 to max_m, embed the signal, compute PCA,
    and determine how many components are significant.

    Returns
    -------
    results : list of dict
        Per-m results with keys: m, eigenvalues, explained_var_ratio,
        n_broken_stick, n_noise_threshold, participation_ratio.
    """
    results = []
    for m in range(2, max_m + 1):
        emb = takens_embedding(data_1d, tau=tau, m=m)  # (m, T')
        X = emb.T  # (T', m) — samples x features

        # Center
        X = X - X.mean(axis=0)

        # Covariance eigendecomposition
        cov = np.cov(X, rowvar=False)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        total_var = eigvals.sum()
        explained = eigvals / total_var

        # Method 1: Broken stick
        bs = broken_stick(m)
        n_bs = int((explained > bs).sum())

        # Method 2: Noise threshold
        # Estimate noise variance as the mean of the smallest half of eigenvalues
        noise_eigvals = eigvals[m // 2:]
        noise_level = noise_eigvals.mean() if len(noise_eigvals) > 0 else 0
        n_noise = int((eigvals > noise_level * 2).sum())  # 2x noise floor

        # Method 3: Participation ratio
        # PR = (Σλ)² / Σλ² — effective number of dimensions
        pr = total_var**2 / (eigvals**2).sum()

        results.append({
            'm': m,
            'eigenvalues': eigvals,
            'explained': explained,
            'n_broken_stick': n_bs,
            'n_noise_threshold': n_noise,
            'participation_ratio': pr,
        })

    return results


def main():
    print('Generating synthetic data...')

    # HD neurons (continuous latent variable, dim=1)
    exp_hd = generate_tuned_selectivity_exp(
        population=[{'name': 'hd', 'count': 200,
                     'features': ['head_direction']}],
        duration=600, fps=5, seed=42, verbose=False)

    # Event neurons (discrete, 3 events)
    exp_ev = generate_tuned_selectivity_exp(
        population=[
            {"name": "event_0", "count": 30, "features": ["event_0"]},
            {"name": "event_1", "count": 30, "features": ["event_1"]},
            {"name": "event_2", "count": 30, "features": ["event_2"]},
        ],
        duration=600, fps=5, seed=42, n_discrete_features=3, verbose=False)

    # Pure noise (no signal)
    rng = np.random.default_rng(42)
    noise_data = rng.standard_normal((10, 3000)) * 0.1

    datasets = [
        ('HD neuron (representative)', exp_hd.calcium.data[0]),
        ('Event neuron (representative)', exp_ev.calcium.data[0]),
        ('Pure noise', noise_data[0]),
    ]

    # Also test: what does a "population-average" signal look like?
    # Average calcium across all HD neurons — this should be nearly constant
    # But the embedded calcium of individual neurons is what matters

    fig, axes = plt.subplots(len(datasets), 2, figsize=(16, 5 * len(datasets)))

    print(f'\n{"Signal":<30} {"tau":>4} {"m_FNN":>6} '
          f'{"m_BS":>5} {"m_noise":>7} {"PR":>6}')
    print('-' * 65)

    for row, (name, signal) in enumerate(datasets):
        # Estimate tau
        tau = estimate_tau(signal, method='exponential_fit')

        # FNN for comparison
        from driada.recurrence.embedding import estimate_embedding_dim
        m_fnn = estimate_embedding_dim(signal, tau=tau, max_dim=15)

        # PCA analysis
        results = signal_dim_pca(signal, tau=tau, max_m=15)

        # Extract summary at m=15 (largest embedding)
        r15 = results[-1]  # m=15
        n_bs = r15['n_broken_stick']
        n_noise = r15['n_noise_threshold']
        pr = r15['participation_ratio']

        print(f'{name:<30} {tau:>4} {m_fnn:>6} '
              f'{n_bs:>5} {n_noise:>7} {pr:>6.1f}')

        # Left plot: eigenvalue spectrum at m=15
        ax = axes[row, 0]
        ax.semilogy(range(1, 16), r15['eigenvalues'], 'ko-', label='Eigenvalues')
        # Broken stick
        bs = broken_stick(15) * r15['eigenvalues'].sum()
        ax.semilogy(range(1, 16), bs, 'r--', alpha=0.7, label='Broken stick')
        # Noise floor
        noise_floor = r15['eigenvalues'][7:].mean()
        ax.axhline(noise_floor, color='blue', ls=':', alpha=0.7,
                   label=f'Noise floor')
        ax.axhline(noise_floor * 2, color='blue', ls='--', alpha=0.5,
                   label=f'2× noise')
        ax.set_xlabel('Component')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(f'{name}\nτ={tau}, FNN→m={m_fnn}, '
                     f'BS→{n_bs}, noise→{n_noise}, PR={pr:.1f}')
        ax.legend(fontsize=8)
        ax.set_xticks(range(1, 16))

        # Right plot: dimension estimates vs embedding m
        ax = axes[row, 1]
        ms = [r['m'] for r in results]
        ax.plot(ms, [r['n_broken_stick'] for r in results],
                'ro-', label='Broken stick')
        ax.plot(ms, [r['n_noise_threshold'] for r in results],
                'bs-', label='Noise threshold')
        ax.plot(ms, [r['participation_ratio'] for r in results],
                'g^-', label='Participation ratio')
        ax.axhline(m_fnn, color='gray', ls=':', label=f'FNN={m_fnn}')
        ax.set_xlabel('Embedding dimension m')
        ax.set_ylabel('Estimated signal dimension')
        ax.set_title('Signal dim vs embedding m')
        ax.legend(fontsize=8)
        ax.set_xticks(ms)

    fig.tight_layout()
    path = OUT / 'fig_signal_dim_pca.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {path}')

    # ================================================================
    # Test: does PCA-based m give good circ_r on HD data?
    # ================================================================
    print('\n--- Validation: circ_r with PCA-based m ---')
    from driada.information.info_base import MultiTimeSeries
    from sklearn.manifold import SpectralEmbedding
    from scipy.stats import pearsonr
    import scipy.sparse as sp

    calcium = exp_hd.calcium.data
    hd_full = exp_hd.dynamic_features['head_direction'].data
    n_frames = calcium.shape[1]

    # Get per-neuron PCA-based m
    pca_ms = []
    for i in range(calcium.shape[0]):
        tau_i = estimate_tau(calcium[i], method='exponential_fit')
        res = signal_dim_pca(calcium[i], tau=tau_i, max_m=10)
        # Use broken stick at m=10
        m_pca = max(2, res[-1]['n_broken_stick'])
        pca_ms.append(m_pca)
    pca_ms = np.array(pca_ms)
    print(f'  PCA-based m: median={int(np.median(pca_ms))}, '
          f'range=[{pca_ms.min()}, {pca_ms.max()}]')

    # Build population recurrence with PCA-based m (use median)
    m_use = int(np.median(pca_ms))
    print(f'  Using m={m_use}')

    mts = MultiTimeSeries(calcium, discrete=False)
    pop_rg = mts.population_recurrence_graph(
        method='mean', k=50, n_jobs=-1,
        tau_method='exponential_fit', max_dim=m_use,
        theiler_window=5,
    )
    mm = pop_rg.adj.toarray() if sp.issparse(pop_rg.adj) else pop_rg.adj
    min_n = mm.shape[0]
    offset = n_frames - min_n
    hd = hd_full[offset:offset + min_n]

    diag = np.abs(np.arange(min_n)[:, None] - np.arange(min_n)[None, :]) < 3
    md = mm.copy()
    md[diag] = 0

    se = SpectralEmbedding(n_components=4, affinity='precomputed',
                           random_state=42)
    coords = se.fit_transform(md)
    best_r = 0
    for d1 in range(4):
        for d2 in range(d1 + 1, 4):
            ea = np.arctan2(coords[:, d2], coords[:, d1])
            rc, _ = pearsonr(np.cos(ea), np.cos(hd))
            rs, _ = pearsonr(np.sin(ea), np.sin(hd))
            cr = np.sqrt(rc**2 + rs**2)
            if cr > best_r:
                best_r = cr

    taus = np.array([ts._recurrence_tau[1] for ts in mts.ts_list])
    dims = np.array([ts._recurrence_graph_cache[0][1] for ts in mts.ts_list])
    emb_win = int(np.median(taus * (dims - 1)))

    print(f'  tau={int(np.median(taus))}, m={m_use}, '
          f'emb_win={emb_win} ({emb_win/5:.1f}s)')
    print(f'  circ_r = {best_r:.3f}')
    print(f'  (reference: m=2 -> 0.81, m=5 -> 0.21)')


if __name__ == '__main__':
    main()
