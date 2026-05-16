# Population Recurrence Analysis from Stacked Neural Recurrence Plots

**No established framework exists for computing individual recurrence plots per neuron, stacking them, and deriving joint population-level metrics — making this a genuinely novel methodological frontier.** A single 2022 paper (Nomura et al.) proposes a "superposed recurrence plot" using OR-aggregation across simulated neurons, but this has never been applied to real recordings. The foundational recurrence analysis literature is mature (1987–present), and multivariate extensions exist (JRP, MdRQA), yet their intersection with modern large-scale neural population recordings remains almost entirely unexplored. The software ecosystem, while rich for univariate and bivariate analysis, lacks the scalable multi-channel tools, neuroscience-pipeline integrations, and statistical frameworks necessary for population-level recurrence analysis at the scale of contemporary calcium imaging or Neuropixels datasets.

---

## Recurrence analysis foundations: from visualization to quantification

The recurrence plot was introduced by **Eckmann, Kamphorst, and Ruelle (1987)** [[1]](#ref-eckmann1987) as a binary matrix $R(i,j) = \Theta(\varepsilon - \|\mathbf{x}_i - \mathbf{x}_j\|)$, where $\mathbf{x}_i$ are phase-space vectors reconstructed via Takens' time-delay embedding. The RP was originally a purely visual tool — a dot at position $(i,j)$ indicates the system's trajectory revisited the same neighborhood at times $i$ and $j$.

**Zbilut and Webber (1992, 1994)** [[2]](#ref-zbilut1992) transformed RPs from visual artifacts into a quantitative methodology by introducing Recurrence Quantification Analysis (RQA). Their key insight was that diagonal line structures in RPs encode deterministic dynamics: two measures — **%Determinism (DET)**, the fraction of recurrence points forming diagonal lines of length ≥ $l_\text{min}$, and **Shannon entropy (ENTR)** of the diagonal line length distribution — became the workhorses of RQA. Additional measures include recurrence rate (RR), average diagonal line length (L), longest diagonal line ($L_\text{max}$, related to the largest Lyapunov exponent), and TREND (detecting nonstationarity). Trulla et al. (1996) [[3]](#ref-trulla1996) demonstrated that windowed RQA could localize bifurcation points in the logistic map.

**Marwan, Wessel, Meyerfeldt, Schirdewan, and Kurths (2002)** [[4]](#ref-marwan2002) completed the RQA toolkit by introducing measures based on vertical line structures: **laminarity (LAM)** and **trapping time (TT)**, which detect laminar phases and chaos-chaos transitions that diagonal-based measures miss. Their comprehensive *Physics Reports* review (Marwan, Romano, Thiel, and Kurths, 2007) [[5]](#ref-marwan2007) remains the definitive reference, covering RP theory, phase-space reconstruction, threshold selection, and applications spanning physiology, geoscience, and economics. A recent trends review (Marwan and Kraemer, 2023) [[6]](#ref-marwan2023) documents ongoing advances including approximative RQA for time series exceeding one million points.

---

## Cross-recurrence, joint recurrence, and multivariate extensions

Three distinct approaches extend recurrence analysis beyond single systems, each with fundamentally different mathematical and interpretive properties.

**Cross-recurrence plots (CRP)**, introduced by Zbilut, Giuliani, and Webber (1998) and formalized by Marwan and Kurths (2002), compare trajectories of two different systems **in the same phase space**: $CR(i,j) = \Theta(\varepsilon - \|\mathbf{x}(i) - \mathbf{y}(j)\|)$. CRPs require identical embedding dimensions but allow different data lengths, producing generally asymmetric matrices. The Line of Synchronization in CRPs reveals time-lag relationships and can resynchronize systems with different time scales. Cross-RQA (CRQA) quantifies coupling strength between two systems but is inherently **pairwise** — scaling to $N$ neurons requires $\binom{N}{2}$ comparisons. See the [CRP Toolbox documentation](http://www.recurrence-plot.tk/crps.php) for formal definitions.

**Joint recurrence plots (JRP)**, introduced by Romano, Thiel, Kurths, and von Bloh (2004) [[7]](#ref-romano2004), take a fundamentally different approach: they compute the **Hadamard (element-wise) product** of individual RPs, requiring simultaneous recurrence in both subsystems independently: $JR(i,j) = \Theta(\varepsilon_x - \|\mathbf{x}(i) - \mathbf{x}(j)\|) \cdot \Theta(\varepsilon_y - \|\mathbf{y}(i) - \mathbf{y}(j)\|)$. Critically, JRPs allow different embedding dimensions for each subsystem and **naturally extend to $N > 2$ systems** by multiplying $N$ individual RP matrices. This extensibility is what makes JRPs the most natural candidate for "stacked RP" population analysis — though as $N$ grows, the joint recurrence rate drops approximately as $\text{RR}^N$ (assuming independence), producing extremely sparse matrices for large neural populations.

**Multidimensional RQA (MdRQA)**, proposed by Wallot, Roepstorff, and Mønster (2016) [[8]](#ref-wallot2016) and tutorialized by Wallot and Leonardi (2018) [[9]](#ref-wallot2018), takes a third path: it **concatenates all channels into a single high-dimensional state vector** $\mathbf{x}_i = (u^1_i, u^2_i, \ldots, u^k_i) \in \mathbb{R}^k$ and computes one RP from the resulting multidimensional trajectory. MdRQA treats multiple time series as dimensions of a single system rather than independent subsystems. It has been applied primarily in psychology and behavioral science (joint action studies, conversation dynamics) — **never to neural population recordings**. Wallot (2019) further extended this to MdCRQA [[10]](#ref-wallot2019) for comparing two multidimensional systems.

The table below captures the critical architectural differences:

| Feature | CRP | JRP | MdRQA |
|---|---|---|---|
| Core operation | Distance between two trajectories in shared space | Hadamard product of individual RPs | Single RP of concatenated dimensions |
| Scaling to $N$ systems | $\binom{N}{2}$ pairwise comparisons | Product of $N$ matrices | Single $N$-dimensional embedding |
| Phase-space dimensions | Must match | Can differ per system | All channels as dimensions |
| Sparsity with $N$ neurons | Unchanged per pair | Exponential sparsification | Curse of dimensionality |
| Interpretation | Trajectory similarity | Simultaneous recurrence | Population state recurrence |

Beyond these three, **multiplex recurrence networks** (Eroglu, Marwan, Stebich, and Kurths, 2018) [[11]](#ref-eroglu2018) offer a fourth strategy: each channel generates its own recurrence network (treating the RP as an adjacency matrix), and these are assembled into a multilayer network. This avoids the curse of dimensionality by analyzing each channel independently and then quantifying inter-layer structural similarity.

---

## Neural applications: a surprisingly sparse landscape

Despite the maturity of RP/RQA methodology, its application to single-neuron and population-level neural recordings is remarkably limited. The vast majority of recurrence studies in neuroscience operate on EEG or LFP signals, not individual neurons.

**The earliest neural RP study** is Kałużny and Tarnecki (1993) [[12]](#ref-kaluzny1993), who computed RPs of interspike interval (ISI) sequences from cerebellar Purkinje cells and red nucleus neurons in anesthetized cats. They found recurring episodes of quasi-deterministic firing patterns and significant deviations from randomness. Novellino et al. (2010) [[13]](#ref-novellino2010) extended RQA to multi-electrode array (MEA) recordings from in vitro cortical networks, tracking developmental maturation of network dynamics through ISI-based RQA on individual channels.

**For calcium imaging**, only one study directly applies recurrence methods. Pérez-Ortega, Guerra et al. (2022) [[14]](#ref-perezortega2022) analyzed ex vivo calcium imaging from striatal brain slices in Parkinson's disease models, computing RQA measures (recurrence rate, determinism, divergence, Markov entropy) on the time series of **population ensemble activation states** — not on individual neuron fluorescence traces. Their pipeline first identified neuronal ensembles via UMAP dimensionality reduction, then treated the sequence of ensemble identity labels as a categorical time series for RP analysis. This is population-level recurrence analysis, but it operates on a derived ensemble-label signal, not on stacked individual-neuron RPs. **No study has applied RQA directly to raw calcium fluorescence traces (ΔF/F) or to deconvolved spike trains from calcium imaging.**

**The closest realization of "stacked RPs"** comes from Nomura, Fujiwara, and Ikeguchi (2022) [[15]](#ref-nomura2022). They computed individual RPs from simulated Izhikevich neuron firing rates receiving a common input, then combined them via **pixel-wise OR (union)**: a pixel $(i,j) = 1$ if any individual neuron RP has a recurrence at that position. The SRP successfully reconstructed the unobserved common input signal. However, this work used only simulated data and aimed at input reconstruction, not population dynamics characterization. It represents the only published attempt at anything resembling stacked-RP population analysis.

**Recurrence networks on neural data** have been applied to EEG (Gao et al., 2013 [[16]](#ref-gao2013); Lopes et al., 2021 [[17]](#ref-lopes2021)) but not to single-neuron recordings. Lopes et al. explicitly note that recurrence networks can infer low-dimensional intrinsic manifolds from brain data, drawing a conceptual parallel to neural manifold methods — but this connection has never been empirically tested on population recordings.

**No study has systematically compared recurrence-based metrics to PCA, GPFA, UMAP, LFADS, or other standard neural population dynamics methods** on the same datasets. This comparison is a critical missing piece.

---

## Software ecosystem: capable tools, absent infrastructure

The recurrence analysis software landscape is mature for univariate and bivariate analysis but lacks the infrastructure for population-scale neural applications.

**[PyRQA](https://pypi.org/project/PyRQA/)** (Python; Rawald, Sips, and Marwan, 2017) [[18]](#ref-pyrqa) is the most scalable option, using **OpenCL for GPU-accelerated computation** that processes time series of 1,000,000+ points (reducing 8-hour computations to ~69 seconds). It supports RP, CRP, JRP, and full RQA/CRQA/JRQA with Euclidean, maximum, and taxicab metrics. Version 8.1.0 (February 2024) is actively maintained. Its primary limitation for population analysis is that it processes one or two time series at a time with no built-in batch or multi-channel mode.

**[CRP Toolbox for MATLAB](https://tocsy.pik-potsdam.de/CRPtoolbox/)** (Marwan; version 5.29, updated 2025) [[19]](#ref-crptoolbox) is the gold-standard reference implementation, offering RP, CRP, JRP, full RQA, recurrence network measures, and phase-space reconstruction tools via both GUI and command-line interfaces. It suffers from significant performance degradation in MATLAB versions after R2014b and has no GPU acceleration, making it impractical for large-scale neural data.

**[pyunicorn](https://www.pik-potsdam.de/~donges/pyunicorn/)** (Donges et al., PIK Potsdam; Python/Cython) [[20]](#ref-pyunicorn) uniquely combines recurrence analysis with **recurrence network analysis**, offering `RecurrencePlot`, `CrossRecurrencePlot`, and `JointRecurrencePlot` classes plus network measures (transitivity, path length, clustering). It supports multi-dimensional input vectors natively and has experimental sparse-RQA mode, but lacks GPU acceleration.

**[RecurrenceAnalysis.jl](https://juliadynamics.github.io/RecurrenceAnalysis.jl/stable/)** (Julia; part of the DynamicalSystems.jl ecosystem by Datseris) [[21]](#ref-recurrencejl) uses sparse matrix storage and Julia's JIT compilation for good performance. It supports RP, CRP, JRP, and full RQA with multiple distance metrics and threshold types.

**[crqa](https://cran.r-project.org/package=crqa)** (R; Coco, Dale, Wallot) implements auto-RQA, CRQA, and MdCRQA for continuous and categorical time series, with `piecewiseRQA` for memory management. Performance is limited beyond ~5,000 time points without calling PyRQA via `reticulate`. **[casnet](https://github.com/FredHasselman/casnet)** (R; Hasselman) [[22]](#ref-casnet) adds multiplex recurrence networks with sparse matrix support.

**MdRQA implementations** exist in MATLAB ([Wallot et al., GitHub](https://github.com/Wallot/MdRQA)), R (via `crqa`), and Python (**[PyMdRQA](https://github.com/furmanlukasz/PyMdRQA)** by Furman, designed for fMRI data) [[23]](#ref-pymdrqa). These handle multivariate time series but assume all channels are dimensions of a single system.

**No recurrence analysis tool integrates with any neuroscience data format or pipeline.** There is zero connectivity to NWB (Neurodata Without Borders), NIX, suite2p, CaImAn, Kilosort, or SpikeInterface. Any population recurrence framework would need to bridge this gap entirely from scratch.

---

## What a population recurrence framework would need to solve

The gap between existing tools and what population-level recurrence analysis demands is substantial and spans methodology, computation, and statistics.

**The fundamental methodological question** is how to aggregate $N$ individual $T \times T$ binary matrices into meaningful population metrics. Four strategies present themselves, each with distinct tradeoffs. Extended JRP (element-wise AND of $N$ matrices) captures simultaneous recurrence across all neurons but becomes exponentially sparse as $N$ grows. The Nomura SRP approach (element-wise OR) captures any-neuron recurrence but loses specificity. MdRQA (single RP of $N$-dimensional trajectory) captures population state recurrence but suffers the curse of dimensionality for large $N$. A novel approach — computing the **population recurrence rate** as the fraction of neurons recurring at each $(i,j)$ pair, yielding a continuous-valued matrix rather than a binary one — has not been proposed in the literature but would naturally generalize between the AND and OR extremes.

**Embedding parameter selection** for neural data is entirely uncharted. Calcium imaging traces have slow dynamics (GCaMP6f decay ~400ms) that will produce artificially thick diagonal lines in RPs, inflating DET and L measures. The appropriate time delay $\tau$ for calcium data (likely 5–15 samples at 30 Hz, i.e., 150–500ms) and embedding dimension $m$ have never been systematically studied. For spike trains, the binary/point-process nature requires either ISI-based RPs, binned spike counts, or edit-distance metrics — Marwan (2023) [[24]](#ref-marwan2023event) specifically addresses the challenges of event-time recurrence analysis and proposes directions for point-process data. No published guidelines exist for choosing binning resolution and its interaction with RP parameters.

**Statistical frameworks are absent.** There are no standardized null models for testing whether population recurrence patterns exceed chance. Twin surrogates (Thiel et al., 2006) preserve individual recurrence structure while destroying phase relationships — they could be adapted for multi-neuron data but require validation. There is no framework for trial-to-trial variability handling: probabilistic latent-variable models (GPFA, LFADS) explicitly model trial variability, but averaging binary RP matrices across trials destroys structural information. Bootstrap or permutation-based approaches for RQA uncertainty quantification on neural data have not been developed.

**Computational scaling is a hard constraint.** For a typical calcium imaging dataset ($N = 500$ neurons, $T = 50{,}000$ frames at 30 Hz over ~28 minutes), each individual RP is a $50{,}000 \times 50{,}000$ binary matrix (~312 MB uncompressed; ~30 MB at 1% sparsity). Five hundred such matrices require ~15 GB of sparse storage. Computing all $N$ RPs requires $N \times T^2 = 1.25 \times 10^{12}$ distance computations. Neuropixels recordings ($N = 1{,}000$+ neurons, $T = 3{,}600{,}000$ at 1-second binning over hours) push this further. PyRQA's GPU acceleration and Marwan and Kraemer's (2023) approximative RQA methods partially address the single-series $T^2$ bottleneck, but no tool handles the multi-channel dimension.

---

## How recurrence methods compare to established population dynamics approaches

Recurrence analysis offers capabilities that existing neural population methods lack, while also having clear deficiencies.

**Unique strengths of recurrence methods** include model-free detection of regime transitions (no training or fitting required), direct quantification of determinism versus stochasticity (DET), estimation of dynamical invariants related to Lyapunov exponents ($L_\text{max}$) and Kolmogorov-Sinai entropy (ENTR), sensitivity to nonlinear coupling that linear coherence and correlation miss, and explicit handling of nonstationarity through windowed RQA and the TREND measure. These are precisely the properties needed to characterize neural population dynamics that may involve nonlinear attractor transitions, metastable states, and regime changes — phenomena increasingly documented in cortical and subcortical circuits.

**What recurrence methods currently cannot do** relative to established approaches: GPFA (Yu et al., 2009) provides smooth single-trial latent trajectories with principled Bayesian inference; LFADS (Pandarinath et al., 2018) denoises single trials through a trained RNN; dPCA (Kobak et al., 2016) [[25]](#ref-dpca) decomposes variance by task parameters in a supervised manner; DPAD (Sani et al., 2024) [[26]](#ref-dpad) dissociates behaviorally-relevant dynamics from other neural variability; MARBLE (Peach et al., 2024) [[27]](#ref-marble) enables cross-animal comparison through geometric deep learning on neural manifolds. Recurrence methods offer none of these capabilities — no generative model, no behavioral prediction, no supervised decomposition, and no alignment across sessions.

The most productive path forward is likely **complementary rather than competitive**. Recurrence metrics could serve as features fed into manifold analysis or as diagnostic tools characterizing the dynamical regime (chaotic, periodic, laminar, transient) of latent trajectories already extracted by GPFA or LFADS. The empirical comparison on shared benchmarks (such as the Neural Latents Benchmark) has never been performed and represents a high-priority gap.

---

## Calcium imaging and spike train data each pose distinct challenges

**Calcium imaging data** presents four specific challenges for recurrence analysis. First, the slow GCaMP decay ($\tau \approx 100{-}500$ ms) acts as a low-pass filter that obscures fast dynamics and creates long autocorrelations, producing thick diagonal structures that artificially inflate DET. Second, fluorescence signals are non-negative and bounded below, potentially distorting phase-space geometry. Third, deconvolution algorithms (OASIS, CASCADE) introduce ringing and threshold artifacts that may produce spurious recurrence structures. Fourth, neuropil contamination introduces shared signals across neurons, inflating cross-recurrence measures. **No published study has addressed any of these issues.** A reasonable starting approach — which would itself constitute novel methodological work — would be to use raw ΔF/F traces with threshold set by fixed recurrence rate (e.g., 5%), Cao's method for embedding dimension, and AMI-based time delay selection, validated against simulated ground-truth data.

**Spike train data** requires fundamentally different choices. Pure spike times are not continuous time series, so standard Euclidean-distance RPs require either ISI transformation, spike count binning, or specialized distance metrics. Marwan (2023) [[24]](#ref-marwan2023event) reviews edit-distance and event-based approaches specifically for point-process recurrence. Binning resolution directly affects what dynamics are visible: fine bins (1–5 ms) capture temporal coding but create very sparse vectors dominated by zeros; coarse bins (50–100 ms) approximate rate coding but destroy fine temporal structure. Kraemer et al. (2022) [[28]](#ref-kraemer2022) showed that τ-recurrence rates from ISI-based RPs yield "spike spectra" revealing dominant frequencies, but this has not been applied to population data. **No unified framework exists for simultaneously analyzing rate and temporal coding through recurrence.**

---

## Recent work (2023–2026) shows growing interest but no breakthroughs in neural populations

Several recent publications advance recurrence methodology without yet crossing into population neuroscience. Marwan and Kraemer (2023) [[6]](#ref-marwan2023) provide a comprehensive trends review identifying key open questions: computational scaling, alternative recurrence definitions for event data, multiscale approaches, and machine-learning integration. Kargarnovin et al. (2024) [[29]](#ref-kargarnovin2024) introduced Multi-Threshold Recurrence Rate Plots (MTRRP) and a "Recurrence Complexity" measure for EEG-based Alzheimer's/frontotemporal dementia classification. A 2024 study applied fuzzy recurrence plots and phase portraits to fMRI functional connectivity [[30]](#ref-fuzzyRP2024), explicitly noting that RPs preserve temporal information lost in dimensionality reduction.

In the competing/complementary space of neural population dynamics, **2024 was a landmark year**: DPAD, DFINE, MARBLE, and nonlinear manifold analyses all demonstrated that nonlinear methods outperform linear dimensionality reduction for neural population data [[26, 27]](#ref-dpad). This validates the core premise motivating population recurrence analysis — that neural dynamics are fundamentally nonlinear — while simultaneously raising the bar for what any new method must demonstrate.

---

## Conclusion

The methodological space of "population recurrence analysis" — computing individual neuron RPs, stacking them, and deriving joint metrics — is genuinely open territory. The theoretical foundations are solid (JRP naturally generalizes to $N$ systems; MdRQA handles multivariate data; multiplex recurrence networks avoid dimensionality issues), yet no one has systematically applied these tools to real multi-neuron recordings from calcium imaging or high-density electrophysiology. The single closest precedent, the Superposed Recurrence Plot (Nomura et al., 2022) [[15]](#ref-nomura2022), operates only on simulated data and uses a specific aggregation strategy (OR) that may not be optimal for characterizing population dynamics.

A viable population recurrence framework would need to address five requirements simultaneously: **(1)** a principled aggregation strategy across individual neuron RPs (moving beyond binary AND/OR to fractional population recurrence rates or weighted combinations); **(2)** validated embedding parameter guidelines for calcium and spike train data; **(3)** GPU-accelerated, sparse-matrix computation scaling to hundreds of neurons and tens of thousands of time points; **(4)** statistical null models (adapted twin surrogates, permutation tests) with proper multiple-comparison correction; and **(5)** integration with the NWB/SpikeInterface/suite2p ecosystem that modern systems neuroscience relies on. The theoretical tools exist in scattered form; assembling them into a coherent, validated framework would constitute a significant methodological contribution to computational neuroscience.

---

## References

<a id="ref-eckmann1987"></a>
**[1]** Eckmann, J.-P., Kamphorst, S. O., & Ruelle, D. (1987). Recurrence plots of dynamical systems. *Europhysics Letters*, 4(9), 973–977. [PDF (IHES)](https://www.ihes.fr/~ruelle/PUBLICATIONS/%5B92%5D.pdf)

<a id="ref-zbilut1992"></a>
**[2]** Zbilut, J. P., & Webber, C. L. (1992). Embeddings and delays as derived from quantification of recurrence plots. *Physics Letters A*, 171(3–4), 199–203. See also: Webber, C. L., & Zbilut, J. P. (1994). [Assessing deterministic structures in physiological systems using recurrence plot strategies](https://link.springer.com/chapter/10.1007/978-0-585-34964-0_8). In *Bioengineering Approaches to Pulmonary Physiology and Medicine*, Springer.

<a id="ref-trulla1996"></a>
**[3]** Trulla, L. L., Giuliani, A., Zbilut, J. P., & Webber, C. L. (1996). Recurrence quantification analysis of the logistic equation with transients. [*Physics Letters A*](https://www.sciencedirect.com/science/article/abs/pii/S0375960196007414), 223(4), 255–260.

<a id="ref-marwan2002"></a>
**[4]** Marwan, N., Wessel, N., Meyerfeldt, U., Schirdewan, A., & Kurths, J. (2002). Recurrence-plot-based measures of complexity and their application to heart-rate-variability data. [*Physical Review E*](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.66.026702), 66(2), 026702.

<a id="ref-marwan2007"></a>
**[5]** Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. [*Physics Reports*](https://www.sciencedirect.com/science/article/abs/pii/S0370157306004066), 438(5–6), 237–329.

<a id="ref-marwan2023"></a>
**[6]** Marwan, N., & Kraemer, K. H. (2023). Trends in recurrence analysis of dynamical systems. [*European Physical Journal Special Topics*](https://link.springer.com/article/10.1140/epjs/s11734-022-00739-8), 232, 5–27. [PDF (PIK)](https://publications.pik-potsdam.de/pubman/item/item_28970_2/component/file_28971/Marwan_2023_s11734-022-00739-8.pdf?mode=download)

<a id="ref-romano2004"></a>
**[7]** Romano, M. C., Thiel, M., Kurths, J., & von Bloh, W. (2004). Multivariate recurrence plots. [*Physics Letters A*](https://www.sciencedirect.com/science/article/abs/pii/S0375960104010953), 330(3–4), 214–223.

<a id="ref-wallot2016"></a>
**[8]** Wallot, S., Roepstorff, A., & Mønster, D. (2016). Multidimensional Recurrence Quantification Analysis (MdRQA) for the analysis of multidimensional time-series: A software implementation in MATLAB and its application to group-level data in joint action. [*Frontiers in Psychology*](https://www.frontiersin.org/articles/10.3389/fpsyg.2016.01835/full), 7, 1835. [Code (GitHub)](https://github.com/Wallot/MdRQA)

<a id="ref-wallot2018"></a>
**[9]** Wallot, S., & Leonardi, G. (2018). Analyzing multivariate dynamics using Cross-Recurrence Quantification Analysis (CRQA), Diagonal-Cross-Recurrence Profiles (DCRP), and Multidimensional Recurrence Quantification Analysis (MdRQA) — A tutorial in R. [*Frontiers in Psychology*](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.02232/full), 9, 2232.

<a id="ref-wallot2019"></a>
**[10]** Wallot, S. (2019). Multidimensional Cross-Recurrence Quantification Analysis (MdCRQA) — A method for quantifying correlation between multivariate time-series. [*Multivariate Behavioral Research*](https://www.tandfonline.com/doi/full/10.1080/00273171.2018.1512846), 54(2), 173–191. [Code (GitHub)](https://github.com/Wallot/MdCRQA)

<a id="ref-eroglu2018"></a>
**[11]** Eroglu, D., Marwan, N., Stebich, M., & Kurths, J. (2018). Multiplex recurrence networks. [*Physical Review E*](https://arxiv.org/abs/2003.03309), 97(1), 012312.

<a id="ref-kaluzny1993"></a>
**[12]** Kałużny, P., & Tarnecki, R. (1993). Recurrence plots of neuronal spike trains. [*Biological Cybernetics*](https://dl.acm.org/doi/10.1007/BF00200812), 68, 527–534.

<a id="ref-novellino2010"></a>
**[13]** Novellino, A., et al. (2010). Recurrence Quantification Analysis of spontaneous electrophysiological activity during development: Characterization of in vitro neuronal networks cultured on Multi Electrode Array chips. [*Advances in Artificial Intelligence*](https://onlinelibrary.wiley.com/doi/10.1155/2010/209254), 2010, 209254.

<a id="ref-perezortega2022"></a>
**[14]** Pérez-Ortega, J., Guerra, A., et al. (2022). Dimensionality reduction and recurrence analysis reveal hidden structures of striatal pathological states. [*Frontiers in Systems Neuroscience*](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2022.975989/full), 16, 975989.

<a id="ref-nomura2022"></a>
**[15]** Nomura, Y., Fujiwara, K., & Ikeguchi, T. (2022). Superposed recurrence plots for reconstructing a common input applied to neurons. [*Physical Review E*](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.106.034205), 106(3), 034205.

<a id="ref-gao2013"></a>
**[16]** Gao, Z., et al. (2013). Recurrence network analysis of the synchronous EEG time series in normal and epileptic brains. [*Cell Biochemistry and Biophysics*](https://link.springer.com/article/10.1007/s12013-012-9452-0), 66, 331–341.

<a id="ref-lopes2021"></a>
**[17]** Lopes, M. A., et al. (2021). Network analysis of time series: Novel approaches to network neuroscience. [*Frontiers in Neuroscience*](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.787068/full), 15, 787068.

<a id="ref-pyrqa"></a>
**[18]** Rawald, T., Sips, M., & Marwan, N. (2017). PyRQA — Conducting Recurrence Quantification Analysis on very long time series efficiently. [arXiv:2402.16853](https://arxiv.org/html/2402.16853v1). [PyPI](https://pypi.org/project/PyRQA/)

<a id="ref-crptoolbox"></a>
**[19]** Marwan, N. CRP Toolbox for MATLAB. [PIK Potsdam](https://tocsy.pik-potsdam.de/CRPtoolbox/) · [SIAM](https://dsweb.siam.org/Software/crp-toolbox)

<a id="ref-pyunicorn"></a>
**[20]** Donges, J. F., et al. pyunicorn — Unified functional network and nonlinear time series analysis. [Documentation](https://www.pik-potsdam.de/~donges/pyunicorn/api/timeseries/recurrence_plot.html)

<a id="ref-recurrencejl"></a>
**[21]** Datseris, G. RecurrenceAnalysis.jl (part of DynamicalSystems.jl). [Documentation](https://juliadynamics.github.io/RecurrenceAnalysis.jl/stable/rplots/)

<a id="ref-casnet"></a>
**[22]** Hasselman, F. casnet: An R toolbox for studying Complex Adaptive Systems and Networks. [GitHub](https://github.com/FredHasselman/casnet)

<a id="ref-pymdrqa"></a>
**[23]** Furman, Ł. PyMdRQA: A Python implementation of Multidimensional Recurrence Quantification Analysis. [GitHub](https://github.com/furmanlukasz/PyMdRQA)

<a id="ref-marwan2023event"></a>
**[24]** Marwan, N. (2023). Challenges and perspectives in recurrence analyses of event time series. [*Frontiers in Applied Mathematics and Statistics*](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1129105/full), 9, 1129105.

<a id="ref-dpca"></a>
**[25]** Kobak, D., et al. (2016). Demixed principal component analysis of neural population data. [*eLife*](https://elifesciences.org/articles/10989), 5, e10989.

<a id="ref-dpad"></a>
**[26]** Sani, O. G., et al. (2024). Dissociative and prioritized modeling of behaviorally relevant neural dynamics using recurrent neural networks. [*Nature Neuroscience*](https://pmc.ncbi.nlm.nih.gov/articles/PMC11452342/).

<a id="ref-marble"></a>
**[27]** Peach, R. L., et al. (2024). MARBLE: Interpretable representations of neural population dynamics using geometric deep learning. [*Nature Methods*](https://www.nature.com/articles/s41592-024-02582-2).

<a id="ref-kraemer2022"></a>
**[28]** Kraemer, K. H., et al. (2022). Spike spectra for recurrences. [*Entropy*](https://www.mdpi.com/1099-4300/24/11/1689), 24(11), 1689.

<a id="ref-kargarnovin2024"></a>
**[29]** Kargarnovin, S., et al. (2024). Multi-Threshold Recurrence Rate Plot: A novel methodology for EEG analysis in Alzheimer's disease and frontotemporal dementia. [*Brain Sciences*](https://www.mdpi.com/2076-3425/14/6/565), 14(6), 565.

<a id="ref-fuzzyRP2024"></a>
**[30]** Exploring nonlinear dynamics in brain functionality through phase portraits and fuzzy recurrence plots (2024). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10888921/).
