# Methodology

This document summarises the experimental pipeline implemented in
`notebooks/TLM_DRU_FINAL.ipynb`. For raw aggregated results see
[`results/summary/`](../results/summary/) and
[`figures/`](../figures/).

## 1. Dataset and task

- **Dataset.** TLM:UAV `Fusion_Data.csv` — 12,253 rows × 17 numeric
  telemetry features + 1 timestamp + 1 label column.
- **Original labels.** `{0, 1, 2, 3, 4}` — `0` = Normal (53.6 %),
  `1`–`4` = distinct fault-injection classes.
- **Primary task.** Binary anomaly detection: `0` vs `{1, 2, 3, 4}`.
- **Secondary task.** Fault-3 one-vs-rest detection.

## 2. Splitting protocol (B2)

Random stratified splits leak across neighbouring time instants and
inflate scores. We adopt a **group-aware, temporal block protocol**:

1. Sort rows by `timestamp`.
2. Partition into **K = 10 contiguous blocks** of equal duration.
3. For each of **10 seeds**, hold out one block as test and use the
   remaining as train (rotating). Optionally a contiguous validation
   block is carved from the train side.
4. The `timestamp` column is used **only** for grouping; it is never a
   feature.

Total: 10 seeds × 10 folds = 100 runs per (model × feature mode).

## 3. Proxy-feature audit

Several telemetry channels correlate trivially with the segment in
which a fault was injected (e.g. cumulative energy, battery state,
GPS-derived quantities). To avoid claiming "anomaly detection" when
the model has actually learned "I know which segment I'm in", we
evaluate three feature modes:

| Mode               | Description                                         |
|--------------------|-----------------------------------------------------|
| `full`             | All 17 numeric features.                            |
| `no_proxy_loose`   | Removes the strongest contextual proxies.           |
| `no_proxy_strict`  | Removes any feature with above-threshold MI to the segment index. |

The audit is logged in `results/eda/proxy_feature_audit.csv` (regenerated
by the notebook).

## 4. Preprocessing

1. **Imputation.** Median per feature (training fold only).
2. **Balancing.** SMOTETomek applied **only to the training fold**.
3. **Feature ranking.** Mutual Information against the binary label,
   computed on the balanced training set.
4. **Encoding for the quantum models.** Top-5 features by MI are
   MinMax-scaled to `[-π, π]`.

## 5. Models

### Classical baselines (5)

Logistic Regression, MLP, Random Forest, SVM-RBF, XGBoost.
All implemented via `scikit-learn` / `xgboost` with default-ish
hyperparameters; no model-specific tuning, to keep the comparison fair.

### Quantum (2)

- **DRU (Data Re-Uploading).**
  5 qubits, 2 layers, `RxRyRz` encoding, ring entanglement, COBYLA
  optimiser. Implemented through the `qdr` package
  (`qiskit-data-reuploading`, pinned in `requirements.txt`).
- **QSVC.** Quantum Support Vector Classifier with a `ZZFeatureMap`
  kernel, as a kernel-method counterpart.

### Hybrid (6 variants)

`XGBoost` trained on `X_q ‖ T(X_q)` where `T` is one of:

- `raw` — identity (control).
- `PCA` — linear control.
- `Poly²` — degree-2 polynomial features (classical non-linear control).
- `RandomRBF` — random Fourier features (classical kernel control).
- `DRU-untrained` — DRU expectation vector with random parameters.
- `DRU-trained` — DRU expectation vector with COBYLA-trained parameters.

The `untrained` and `PCA`/`Poly²`/`RBF` variants are **paired controls**:
they isolate "DRU contributes signal beyond what a linear or random
non-linear lift would give".

## 6. Metrics

For each run we record: F1 macro, balanced accuracy, MCC, ROC-AUC
(one-vs-rest), PR-AUC, detection rate on the anomaly class, and false
alarm rate on class 0 (FAR). Aggregations are mean ± std over the 100
runs per cell.

## 7. Headline result

Only the **trained-DRU hybrid** improves its F1 macro when going from
`full` → `no_proxy_strict` (+0.05) and attains the lowest FAR under
strict evaluation. We report this as an **incremental, reproducible
quantum-enhanced hybrid benefit**, not a categorical "quantum
advantage".
