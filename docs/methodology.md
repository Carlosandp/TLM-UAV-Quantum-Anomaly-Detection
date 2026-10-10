# Methodology

This document describes what `notebooks/TLM_DRU_FINAL.ipynb` actually does
and what the revision analyses in `revision/` add. It replaces an earlier
version that described a different pipeline (a 12,253-row fused table with 17
features and 100 runs per cell), which is not the pipeline that produced the
stored results.

## 1. Data and task

- **Source.** The raw per-sensor logs of TLM:UAV (`ATT`, `BARO`, `BAT`,
  `CTUN`, `MAG`, `MOTB`, `PSCD`, `RATE`, `XKF1`, `GPS`, `IMU`, `VIBE`). The
  released `Fusion_Data.csv` is **not** used: it contains two pairs of exactly
  duplicated columns (`ErrRP`/`ErrYaw`, `MagY`/`MagZ`) that are not duplicates
  in the raw files.
- **Merge.** Exact join on `TimeUS` for the 10 Hz streams; `merge_asof` for
  GPS, IMU, and VIBE; explicit renames for name collisions; per-row label by
  majority vote over the sensor files (no ties occurred).
- **Result.** 4,817 rows, 81 numeric variables, 72 after dropping nine
  constant columns. `TimeUS` is used for ordering only and is never an input.
- **Structure.** One log of 506 s on a single clock. Two logging gaps (16.3 s
  and 8.4 s). Four contiguous fault intervals, one per fault type:

  | Label | Fault (checked against the signals) | Interval (s) | Rows |
  |---|---|---|---|
  | 1 | GPS (satellite count drops from 10 to 3) | 92.0–152.7 | 608 |
  | 2 | Accelerometer (all three channels constant) | 226.0–285.3 | 594 |
  | 3 | Engine (on the ground, throttle saturated, zero speed) | 335.0–383.1 | 482 |
  | 4 | Remote control (descent) | 430.9–506.1 | 670 |

- **Task.** Binary: fault (labels 1–4) vs. nominal (label 0). A multiclass
  task cannot be evaluated on unseen events because each fault type has a
  single interval.
- **Scope.** The faults are simulated component faults. No attacker is
  modeled. The benchmark does not test the detection of spoofing or
  intrusions.

## 2. Evaluation protocol: group-aware blocked temporal evaluation

1. Sort rows by `TimeUS` and cut them into **10 contiguous blocks** of equal
   row count (about 48 s each).
2. For each of **10 seeds** (0–9), a two-stage `GroupShuffleSplit` assigns
   whole blocks to training (6), validation (2), and test (2).
3. One run per (model, input set, seed): 10 runs per cell.

What this protocol does **not** guarantee:

- Neighboring blocks are correlated; in every partition a test block borders
  a training block.
- Blocks are assigned at random, so training contains rows recorded after
  test rows in all ten partitions.
- Every fault interval straddles two neighboring blocks, so a test block
  often continues a fault that training has seen.
- The ten seeds reshuffle the same blocks of the same log. Their spread is a
  sensitivity measure, not variability across independent flights.

The validation blocks are held out but are not used for tuning.

## 3. Input sets

| Mode | Variables | Question it answers |
|---|---|---|
| `full` | 72 | What does a detector achieve with all context? |
| `no_proxy_loose` | 63 | What remains without the nine variables most suspected of acting as proxies (cumulative energy/current, battery-resistance estimates, IMU temperature, barometer offset, controller outputs)? |
| `no_proxy_strict` | 22 | What can be detected from attitude, rates, inertial, magnetic, and vibration signals alone? |

The lists are fixed by hand in the notebook (they are not selected by a
mutual-information threshold). The excluded variables are not claimed to
leak; battery, altitude, position, and controller variables are legitimate
inputs when they are available and trustworthy. The exclusions are a design
choice for asking the three questions above.

## 4. Preprocessing (fitted on training rows only, per seed)

1. `RobustScaler`.
2. `SMOTETomek` class rebalancing.
3. Mutual-information ranking of the variables on the rebalanced training
   rows.
4. For the circuit and the hybrids: the top five variables, MinMax-scaled to
   `[-π, π]`.

## 5. Models

- **Classical (5).** Logistic regression, SVM-RBF, random forest, MLP,
  XGBoost; library defaults; all variables of the input set; all rebalanced
  training rows.
- **Circuit (DRU).** 5 qubits, 2 layers, `RxRyRz` encoding, ring
  entanglement, 30 weights, COBYLA with 200 iterations, exact state-vector
  simulation. Trained on subset A: 800 rows (400 per class) of the rebalanced
  training rows.
- **Hybrids (6).** XGBoost (100 trees, depth 6, learning rate 0.1) on the five
  selected variables plus: nothing (`XGB-raw`), PCA, degree-2 polynomial
  terms, random RBF features, the output of an untrained DRU, or the output
  of the trained DRU. Trained on subset B: 800 rows disjoint from A.
- **QSVC.** Implemented in the notebook but disabled (`RUN_QSVC = False`); no
  QSVC results exist.

The trained-DRU hybrid draws on the labels of A and B (1,600 rows); the other
hybrids draw on B only.

## 6. Metrics

F1 macro, balanced accuracy, MCC, ROC AUC, PR AUC, and two error rates.

> **Naming caveat.** In the stored per-run files, `far_class0` is the share of
> fault samples classified as normal, i.e. the **miss rate** (it equals
> `1 - dr_anomaly`). The conventional **false-alarm rate** (nominal samples
> flagged as faults) is `2 * far_macro - far_class0` for the binary task. The
> revision outputs report both under their standard names (`far`, `miss`).

## 7. What the results support

See the README. In short: the choice of split and of input variables changes
the scores far more than the choice of model; the quantum-classical models
perform within the spread of the classical ones; no quantum advantage is
demonstrated; error rates are far from operational values.

## 8. Revision analyses (`revision/`)

| Script | Analysis |
|---|---|
| `rev_classical.py` | Reproduction of the stored classical results; block/partition structure; random-row split; time-gap ("embargo") sensitivity; past-only training; segment hold-out; results without SMOTETomek; synthetic-row diagnostics; conventional false-alarm rate for all models; fairness decomposition and label-informed classical controls |
| `rev_events.py` | Detection rate by whether the fault interval of a test sample also contributed rows to training |
| `rev_dru.py d1` | One stored circuit fit repeated: reproduction check, circuit-evaluation count, wall-clock time, inference latency |
| `rev_dru.py d2` | Circuit and hybrids without synthetic rows (physical-only inputs, ten fits) |
| `rev_figures.py` | Figures of the revised manuscript |
| `rev_supplement_tables.py` | Tables of the supplementary material |
