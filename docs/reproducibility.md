# Reproducibility

## 1. Environment

- **Python.** 3.10 or newer (tested on 3.12).
- **OS.** Tested on macOS 14+. Linux should work identically; on
  Windows, prefer WSL2 for matching SciPy/qiskit-aer behaviour.
- **Dependencies.** Pinned in [`requirements.txt`](../requirements.txt).
  The DRU package is installed from a **specific commit SHA** of
  `Carlosandp/qiskit-data-reuploading` to guarantee bit-for-bit
  reproducibility of circuit construction.

### Install

```bash
git clone https://github.com/Carlosandp/TLM-UAV-Quantum-Anomaly-Detection.git
cd TLM-UAV-Quantum-Anomaly-Detection
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Verify

```bash
python examples/minimal_example.py
```

Expected output: a trained DRU classifier reporting a training accuracy
≥ 0.85 on a synthetic linearly-separable dataset.

## 2. Path configuration

The notebook resolves all data, results, and figure locations from a
single root, controlled by the `TLM_UAV_ROOT` environment variable.
If it is unset, the current working directory is used.

```python
from pathlib import Path
import os

ROOT = Path(os.environ.get("TLM_UAV_ROOT", Path.cwd()))
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
```

To reproduce against an out-of-tree data location:

```bash
export TLM_UAV_ROOT=/path/to/your/working/copy
jupyter notebook notebooks/TLM_DRU_FINAL.ipynb
```

The notebook expects `data/dataset/Fusion_Data.csv` under `ROOT`. See
[`data/README.md`](../data/README.md) for the file specification and
SHA-256 checksum.

## 3. Determinism

- **Seeds.** 10 seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`. Set via
  `numpy.random.seed`, `random.seed`, scikit-learn `random_state`, and
  the `seed_simulator` argument of `qiskit-aer`.
- **SMOTETomek.** Each (seed, fold) pair seeds the resampler
  independently.
- **DRU optimiser (COBYLA).** Deterministic given the parameter init,
  which is seeded.
- **XGBoost.** `n_jobs=1` for determinism (the notebook sets this).

Even with all seeds fixed, results may differ by ~1e-3 across
platforms due to BLAS implementation differences. This does not change
the qualitative ranking of models.

## 4. Runtime expectations

Approximate wall-clock for the full notebook on a 2023 MacBook Pro
(M2 Pro, 16 GB):

| Stage                                  | Time           |
|----------------------------------------|----------------|
| EDA + proxy audit                      | ~1 min         |
| Classical baselines × 5 × 3 modes      | ~10 min        |
| DRU (5 qubits, 2 layers) × 10 seeds    | ~90 min        |
| QSVC × 10 seeds                        | ~30 min        |
| Hybrid heads × 6 × 10 seeds            | ~25 min        |
| Aggregation + figures                  | ~1 min         |
| **Total**                              | **~2.5–3 h**   |

The DRU stage dominates and runs on the Aer state-vector simulator.
GPU is not used. Memory peak is ~3 GB.

## 5. Outputs

Running the notebook end-to-end regenerates:

- `results/eda/*.csv`
- `results/classical/*.csv`
- `results/quantum/*.csv`
- `results/hybrid/*.csv`
- `results/summary/final_comparison_*.csv`
- `results/summary/quantum_advantage_report.md`
- `figures/*.png`

The committed copies of these files are the reference output for the
manuscript.
