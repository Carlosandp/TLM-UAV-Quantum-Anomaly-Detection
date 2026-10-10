# Reproducibility

## 1. Environment

The stored results and the revision analyses were produced with:

| Package | Version |
|---|---|
| Python | 3.12.10 |
| numpy | 2.4.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| imbalanced-learn | 0.14.1 |
| xgboost | 3.2.0 |
| scipy | 1.17.1 |
| qiskit | 2.3.0 |
| qiskit-data-reuploading (`qdr`) | 0.1.0 (commit pinned in `requirements.txt`) |

Host: Apple M5 laptop, 10 cores, 16 GB RAM, macOS 27.

```bash
git clone https://github.com/Carlosandp/TLM-UAV-Quantum-Anomaly-Detection.git
cd TLM-UAV-Quantum-Anomaly-Detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python examples/minimal_example.py     # smoke test on synthetic data
```

## 2. Data and paths

The pipeline reads the **raw per-sensor folders** of TLM:UAV, not
`Fusion_Data.csv` (see [`data/README.md`](../data/README.md)).

| Variable | Meaning | Default |
|---|---|---|
| `TLM_UAV_ROOT` | repository root | current directory (notebook) / detected (scripts) |
| `TLM_DATASET_DIR` | folder containing `ATT/`, `BARO/`, `BAT/`, ... | `<root>/data/dataset` |
| `TLM_RESULT_DIR` | notebook output folder | `<root>/results` |
| `TLM_REV_DIR` | revision output folder | `<root>/revision/results` |

## 3. Main pipeline

```bash
jupyter notebook notebooks/TLM_DRU_FINAL.ipynb
```

- Seeds 0–9 select the block assignment; every estimator receives an
  explicit `random_state`.
- 150 classical runs (about 1 min), 30 circuit fits (225–241 s each, 117 min
  in total), 180 hybrid runs (about 2 min). The quantum-kernel section is
  disabled.
- The notebook writes `results/*` and overwrites `results/experiment.log`.

## 4. Revision analyses

```bash
export TLM_DATASET_DIR=/path/to/dataset
cd revision/scripts
python rev_classical.py
python rev_events.py
python rev_dru.py d1
python rev_dru.py d2
python rev_figures.py
python rev_supplement_tables.py
```

These scripts execute the notebook's own data-building, split, and
preparation cells unmodified and never overwrite `results/`. Measured run
times and exit codes of the last complete run are in
[`revision/results/run_log.txt`](../revision/results/run_log.txt).

## 5. What has been verified

- **Classical results.** Rerunning the notebook code reproduces all 150 stored
  classical runs exactly (maximum absolute difference in F1, ROC AUC, and
  `far_class0`: about 1e-16).
- **Circuit.** The stored fit for seed 0, physical-only inputs, was repeated
  and reproduced exactly (F1 0.1362, ROC AUC 0.0190).
- **Hybrid control.** The stored `XGB-raw` results are reproduced exactly (30
  of 30 runs).
- **Not re-run.** The remaining 29 stored circuit fits and the hybrid
  variants that depend on them were not repeated; they are taken from the
  stored files.

Cross-platform differences in BLAS may change results slightly on other
machines.
