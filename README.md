# TLM-UAV Quantum Anomaly Detection

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/qiskit-2.x-purple)](https://qiskit.org/)

> **Reproducible code, figures, and results for the study on
> quantum machine learning applied to UAV cyber-physical anomaly detection.**

This repository contains the code, figures, and experimental results used in
the study on quantum machine learning for UAV cyber-physical anomaly
detection. **The manuscript itself is not included** in this repository
(neither the LaTeX sources nor the compiled PDF are distributed here).

---

## 1. Project description

UAVs are cyber-physical systems whose telemetry can be poisoned, spoofed, or
degraded by an adversary. This project benchmarks **classical, quantum, and
hybrid quantum-classical** anomaly detectors on the multi-sensor
[TLM:UAV](https://doi.org/10.3390/app13074301) software-in-the-loop dataset
under a **leakage-free, group-aware temporal protocol** (B2) with an
explicit **proxy-feature audit** in three modes (`full` / `loose` /
`strict`).

The headline finding: a hybrid `XGBoost + Data Re-Uploading (DRU)`
classifier is the only model whose F1 macro **improves** when contextual
proxies are removed (`+0.05` from `full` to `strict`) and that achieves the
**lowest false-alarm rate** under proxy-free evaluation. The standalone DRU
is not competitive with the strongest classical baseline; we report this
as an *incremental but reproducible quantum-enhanced hybrid benefit* rather
than a categorical "quantum advantage".

## 2. Scientific motivation

Two methodological hazards inflate reported QML scores in the literature:

1. **Temporal data leakage** — random stratified splits mix samples from
   neighbouring time instants between train and test.
2. **Contextual proxy features** — cumulative energy, battery state, and
   GPS trajectory correlate trivially with the temporal segment in which a
   fault was injected.

The B2 protocol (10 contiguous `TimeUS` blocks, 10 seeds) and the
three-mode feature audit jointly neutralise both hazards.

## 3. Methodology summary

| Component       | Choice                                                        |
|-----------------|---------------------------------------------------------------|
| Dataset         | TLM:UAV (4,817 samples, ~60 numeric features)                 |
| Task            | Binary anomaly detection (+ Fault-3 secondary)                |
| Split           | Group-aware, K=10 contiguous `TimeUS` blocks, 10 seeds        |
| Balancing       | SMOTETomek on training fold only                              |
| Feature ranking | Mutual Information on balanced training set                   |
| Encoding        | Top-5 features → MinMax to `[-π, π]`                          |
| DRU circuit     | 5 qubits, 2 layers, RxRyRz encoding, ring entanglement        |
| Optimiser       | COBYLA                                                        |
| Hybrid head     | XGBoost on `X_q ‖ T(X_q)` for T ∈ {raw, PCA, Poly², RBF, DRU} |

See [`docs/methodology.md`](docs/methodology.md) for the full pipeline.

## 4. Main results

Mean ± std over 10 seeds, B2 protocol, binary task
(see [`results/summary/final_comparison_aggregated.csv`](results/summary/final_comparison_aggregated.csv)):

| Model                | F1 (full) | F1 (strict) | FAR (strict) ↓ |
|----------------------|-----------|-------------|----------------|
| Logistic Regression  | **0.650** | 0.506       | 0.520          |
| Random Forest        | 0.564     | 0.545       | 0.480          |
| XGBoost              | 0.521     | 0.524       | 0.513          |
| Standalone DRU       | 0.568     | 0.510       | 0.509          |
| **XGB + DRU-trained**| 0.509     | **0.561**   | **0.451**      |

Only the trained-DRU hybrid **improves** under strict (proxy-free) evaluation.

## 5. Installation

```bash
git clone https://github.com/Carlosandp/TLM-UAV-Quantum-Anomaly-Detection.git
cd TLM-UAV-Quantum-Anomaly-Detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The Data-Re-Uploading classifier is pulled in as an external dependency
(`qiskit-data-reuploading`, pinned to a specific upstream commit SHA in
`requirements.txt`); no local source copy is maintained in this repo.

## 6. Quick example

```python
import numpy as np
from qdr import DataReuploadingClassifier

X = np.random.uniform(-np.pi, np.pi, size=(200, 5))
y = (X.sum(axis=1) > 0).astype(int)

clf = DataReuploadingClassifier(n_qubits=5, n_layers=2, max_iter=100, seed=0)
clf.fit(X, y)
print("Accuracy:", clf.score(X, y))
```

A runnable version lives at [`examples/minimal_example.py`](examples/minimal_example.py).

## 7. Reproducing the full pipeline

Open and run the end-to-end notebook:

```bash
jupyter notebook notebooks/TLM_DRU_FINAL.ipynb
```

Optionally, set `TLM_UAV_ROOT` to point at an out-of-tree working copy
(see [`docs/reproducibility.md`](docs/reproducibility.md) for the full
recipe, hardware notes, seeds, and runtime expectations).

The full `Fusion_Data.csv` is **not** redistributed in this repo — see
[`data/README.md`](data/README.md) for the source, expected schema, and
SHA-256 checksum. A 100-row sample is provided at
[`data/sample/Fusion_Data_sample.csv`](data/sample/Fusion_Data_sample.csv)
for smoke-testing.

## 8. Repository structure

```
TLM-UAV-Quantum-Anomaly-Detection/
├── notebooks/TLM_DRU_FINAL.ipynb     # end-to-end pipeline
├── examples/minimal_example.py       # synthetic-data smoke test
├── data/
│   ├── README.md                     # how to obtain Fusion_Data.csv
│   └── sample/Fusion_Data_sample.csv # 100-row schema sample
├── figures/                          # publication-quality PNGs
├── results/summary/                  # aggregated CSVs & advantage report
├── docs/
│   ├── methodology.md
│   └── reproducibility.md
├── requirements.txt
├── CITATION.cff
├── LICENSE                           # CC BY-NC-SA 4.0
├── .gitignore
└── README.md
```

## 9. How to cite

If you use this code or results, please cite both this repository
(see [`CITATION.cff`](CITATION.cff)) and the TLM:UAV dataset:

> Yang, T., Lu, Y., Deng, H., Chen, J., & Tang, X. (2023).
> *Acquisition and Processing of UAV Fault Data Based on Time Line
> Modeling Method.* Applied Sciences, 13(7), 4301.
> https://doi.org/10.3390/app13074301

## 10. License

This project is released under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)](LICENSE) license.

## 11. Contact

- Carlos A. Durán Paredes — `caduran@unicauca.edu.co`
- German Darío Díaz — `germandiaz@unicauca.edu.co`
- Issues and pull requests welcome at the repository tracker.
