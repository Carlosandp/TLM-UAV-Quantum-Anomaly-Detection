# TLM-UAV Quantum Anomaly Detection

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/qiskit-2.x-purple)](https://qiskit.org/)

> **Code, figures, and results for a case study on how to evaluate
> machine-learning anomaly detectors for UAV telemetry, with a simulated
> quantum circuit and a hybrid quantum-classical model as the worked example.**

The manuscript itself is not included in this repository (neither the LaTeX
sources nor the compiled PDF).

> **Revision notice (October 2026).** An audit carried out for the revision of
> the manuscript corrected three statements made in earlier versions of this
> repository. See [Corrections](#corrections-to-earlier-versions) below. The
> stored result files are unchanged; what changed is how they are labeled and
> interpreted, plus new analyses in [`revision/`](revision/).

---

## 1. What this project is

The study uses the public [TLM:UAV](https://doi.org/10.3390/app13074301)
benchmark of **simulated** UAV component faults to show how much an evaluation
design can inflate the apparent performance of an anomaly detector, and how to
check for it:

- **Temporal leakage.** Consecutive telemetry samples are nearly identical. A
  random split of rows puts neighbors of every test sample in the training
  set.
- **Proxy variables.** Variables that track mission progress (consumed energy,
  altitude, position) identify *when* a fault was injected rather than *what*
  it looks like.
- **Fair comparison.** A new model family (here, a data re-uploading quantum
  circuit and a hybrid XGBoost + circuit model) needs controls that receive
  the same information.

The quantum models are an illustration. The results **do not demonstrate a
quantum advantage**, and the benchmark contains **simulated faults, not
cyberattacks**.

## 2. Main findings

All numbers come from one simulated flight log of 506 s (4,817 rows, about
10 Hz) that contains **four fault intervals, one per fault type**.

| Check | Result |
|---|---|
| Random split of rows vs. held-out time blocks (mean F1 of five classical models) | 0.98 vs. 0.59 with all 72 variables; 0.91 vs. 0.51 with 22 physical-only variables |
| Removing training rows within 15 s / 30 s of any test row (all variables) | F1 falls from 0.59 to 0.49 / 0.34 |
| Fault samples detected when part of the same fault interval is in training vs. when it is not (physical-only, five classical models) | 84 % vs. 5 % |
| Training data recorded later than the earliest test sample | present in all 10 partitions |
| Effect of removing SMOTETomek | mean F1 changes by at most 0.02 |

The ten "seeds" reshuffle the same ten time blocks of the same log. Their
spread measures **sensitivity to the partition**. They are not ten independent
flights.

### Blocked evaluation, binary task (mean ± std over the 10 partitions)

| Model | F1 (all) | F1 (physical) | ROC AUC (physical) | False-alarm rate (physical) | Miss rate (physical) |
|---|---|---|---|---|---|
| Logistic regression | 0.65 ± 0.17 | 0.51 ± 0.14 | 0.55 ± 0.15 | 0.31 ± 0.30 | 0.52 ± 0.32 |
| SVM (RBF) | 0.58 ± 0.18 | 0.46 ± 0.20 | 0.58 ± 0.19 | 0.32 ± 0.30 | 0.55 ± 0.41 |
| MLP | 0.64 ± 0.16 | 0.53 ± 0.14 | 0.62 ± 0.30 | 0.26 ± 0.28 | 0.51 ± 0.36 |
| Random forest | 0.56 ± 0.13 | 0.54 ± 0.15 | 0.73 ± 0.11 | 0.27 ± 0.29 | 0.48 ± 0.36 |
| XGBoost | 0.52 ± 0.14 | 0.52 ± 0.13 | 0.67 ± 0.15 | 0.28 ± 0.30 | 0.51 ± 0.34 |
| DRU circuit alone (simulated) | 0.57 ± 0.22 | 0.51 ± 0.20 | 0.56 ± 0.27 | 0.42 ± 0.28 | 0.51 ± 0.19 |
| XGB on 5 variables (control) | 0.35 ± 0.10 | 0.52 ± 0.15 | 0.56 ± 0.22 | 0.26 ± 0.30 | 0.53 ± 0.32 |
| XGB + PCA (control) | 0.37 ± 0.07 | 0.53 ± 0.19 | 0.60 ± 0.23 | 0.36 ± 0.30 | 0.46 ± 0.39 |
| XGB + polynomial (control) | 0.40 ± 0.13 | 0.53 ± 0.15 | 0.67 ± 0.16 | 0.28 ± 0.29 | 0.52 ± 0.33 |
| XGB + random RBF (control) | 0.50 ± 0.18 | 0.51 ± 0.14 | 0.56 ± 0.22 | 0.34 ± 0.29 | 0.51 ± 0.35 |
| XGB + untrained DRU (control) | 0.34 ± 0.14 | 0.54 ± 0.16 | 0.57 ± 0.23 | 0.29 ± 0.32 | 0.49 ± 0.35 |
| XGB + trained DRU (hybrid) | 0.51 ± 0.22 | 0.56 ± 0.14 | 0.61 ± 0.17 | 0.34 ± 0.30 | 0.45 ± 0.27 |

*all*: 72 variables. *physical*: 22 attitude, rate, inertial, magnetic, and
vibration variables. **False-alarm rate**: nominal samples flagged as faults.
**Miss rate**: fault samples labeled nominal. Default decision thresholds, no
tuning. Complete tables: [`revision/supplementary_material.pdf`](revision/supplementary_material.pdf).

How to read this table:

- The hybrid with the trained circuit has the highest mean F1 and the lowest
  mean miss rate with physical-only inputs. The random forest has the highest
  ROC AUC. Several models have a lower false-alarm rate than the hybrid.
- Differences between means are small compared with the spread. Partition by
  partition, the hybrid beats the untrained-circuit control in 5 of 10 cases
  and the random forest in 4 of 10.
- False-alarm rates of 0.26–0.42 and miss rates near 0.5 are far from
  operational values. No model here is a deployable detector.
- The circuit and the hybrids use 5 selected variables and 800 training rows;
  the classical baselines use all variables and all training rows. The
  comparison is not like for like (see the supplementary material, Section S5).
- All circuit results are noise-free state-vector simulations on a classical
  computer. Nothing was run on quantum hardware.

## Corrections to earlier versions

1. **"Leakage-free" was too strong.** Holding out whole time blocks is better
   than splitting rows at random, but neighboring blocks remain correlated and
   training includes data recorded after test data. The protocol is now called
   a *group-aware blocked temporal evaluation*.
2. **The column reported as "FAR" was the miss rate.** In the stored result
   files, `far_class0` is the fraction of fault samples classified as normal.
   The conventional false-alarm rate is recovered as
   `2 * far_macro - far_class0` (binary task) and is what the table above
   reports. The earlier statement that the hybrid has the lowest false-alarm
   rate does not hold.
3. **"The only model that improves from `full` to `strict`" was wrong.** All
   six five-variable XGBoost variants improve, most of them by more than the
   trained-circuit hybrid. The cause is that, with all variables, the five
   selected variables are accumulating quantities that fall outside the
   training range at test time, and the plain five-variable XGBoost predicts a
   single class in 7 of 10 partitions.

In addition, the secondary "Fault-3" experiment rests on a **single** valid
partition (9 of 10 random assignments were skipped because a split had no
fault sample) and is an anecdote, not an estimate. The file
[`results/summary/quantum_advantage_report.md`](results/summary/quantum_advantage_report.md)
is kept for traceability and carries a notice.

## 3. Method summary

| Component       | Choice                                                        |
|-----------------|---------------------------------------------------------------|
| Data            | TLM:UAV raw sensor logs, re-merged on the autopilot timestamp: 4,817 rows, 72 numeric variables (the released `Fusion_Data.csv` is **not** used) |
| Task            | Binary: fault vs. nominal                                     |
| Split           | 10 contiguous time blocks; whole blocks assigned to train/validation/test (6/2/2); 10 random assignments |
| Input sets      | `full` (72), `no_proxy_loose` (63), `no_proxy_strict` (22, physical-only) |
| Balancing       | SMOTETomek on training rows only (results without it in `revision/`) |
| Classical       | Logistic regression, SVM-RBF, random forest, MLP, XGBoost (defaults) |
| Circuit (DRU)   | 5 qubits, 2 layers, RxRyRz encoding, ring entanglement, COBYLA, 30 weights, exact simulation |
| Hybrids         | XGBoost on the 5 selected variables plus {nothing, PCA, polynomial, random RBF, untrained DRU output, trained DRU output} |

See [`docs/methodology.md`](docs/methodology.md).

## 4. Installation

```bash
git clone https://github.com/Carlosandp/TLM-UAV-Quantum-Anomaly-Detection.git
cd TLM-UAV-Quantum-Anomaly-Detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The data re-uploading classifier is an external dependency
(`qiskit-data-reuploading`, pinned to a commit in `requirements.txt`).

## 5. Reproducing

Main pipeline (about two hours, dominated by 30 simulated circuit fits):

```bash
jupyter notebook notebooks/TLM_DRU_FINAL.ipynb
```

Revision analyses (split design, fault events, resampling, fairness, cost):

```bash
export TLM_DATASET_DIR=/path/to/TLM_UAV/dataset   # folder with ATT/, BARO/, ...
cd revision/scripts
python rev_classical.py      # ~7 min
python rev_events.py         # ~1 min
python rev_dru.py d1         # ~4 min, one circuit fit
python rev_dru.py d2         # ~15 min, ten circuit fits in parallel
python rev_figures.py
python rev_supplement_tables.py
```

Details, versions, and expected outputs: [`docs/reproducibility.md`](docs/reproducibility.md)
and [`revision/README.md`](revision/README.md). The raw TLM:UAV files are not
redistributed here; see [`data/README.md`](data/README.md).

## 6. Repository structure

```
├── notebooks/TLM_DRU_FINAL.ipynb     # main pipeline
├── results/                          # stored per-run and aggregated results of the main pipeline
├── figures/                          # figures of the original submission (superseded by revision/figures)
├── revision/
│   ├── scripts/                      # revision analyses
│   ├── results/                      # their outputs (CSV, JSON, logs)
│   ├── figures/                      # figures of the revised manuscript
│   └── supplementary_material.pdf    # complete tables
├── docs/methodology.md, docs/reproducibility.md
├── data/README.md, data/sample/
├── examples/minimal_example.py
├── requirements.txt, CITATION.cff, LICENSE
```

## 7. How to cite

Please cite this repository (see [`CITATION.cff`](CITATION.cff)) and the
TLM:UAV data set:

> Yang, T., Lu, Y., Deng, H., Chen, J., & Tang, X. (2023).
> *Acquisition and Processing of UAV Fault Data Based on Time Line
> Modeling Method.* Applied Sciences, 13(7), 4301.
> https://doi.org/10.3390/app13074301

## 8. License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)](LICENSE).

## 9. Contact

- Carlos A. Durán Paredes — `caduran@unicauca.edu.co`
- German Darío Díaz — `germandiaz@unicauca.edu.co`
- Issues and pull requests are welcome.
