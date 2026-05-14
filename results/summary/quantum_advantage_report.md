# Quantum Advantage Report — TLM:UAV

**Protocol:** B2 group-aware, K=10 blocks, 10 seeds, binary task.
**Feature modes:** full, no_proxy_loose, no_proxy_strict.
**Models compared:** classical (5), quantum (DRU, QSVC), hybrid (6 variants including paired controls).

## 1. Headline metrics (mean ± std across 10 seeds)

### F1 macro

| Model | full | loose | strict |
|---|---|---|---|
| LogReg (classical) | 0.650 ± 0.170 | 0.649 ± 0.173 | 0.506 ± 0.142 |
| MLP (classical) | 0.641 ± 0.157 | 0.651 ± 0.172 | 0.528 ± 0.139 |
| RandomForest (classical) | 0.564 ± 0.131 | 0.566 ± 0.139 | 0.545 ± 0.148 |
| SVM-RBF (classical) | 0.577 ± 0.181 | 0.584 ± 0.183 | 0.462 ± 0.205 |
| XGBoost (classical) | 0.521 ± 0.136 | 0.538 ± 0.138 | 0.524 ± 0.127 |
| XGB-DRU-trained (hybrid) | 0.509 ± 0.221 | 0.529 ± 0.185 | 0.561 ± 0.137 |
| XGB-DRU-untrained (hybrid) | 0.345 ± 0.145 | 0.471 ± 0.200 | 0.540 ± 0.160 |
| XGB-PCA (hybrid) | 0.368 ± 0.066 | 0.489 ± 0.229 | 0.532 ± 0.185 |
| XGB-Poly2 (hybrid) | 0.404 ± 0.135 | 0.496 ± 0.161 | 0.531 ± 0.153 |
| XGB-RandomRBF (hybrid) | 0.498 ± 0.177 | 0.444 ± 0.155 | 0.509 ± 0.144 |
| XGB-raw (hybrid) | 0.345 ± 0.098 | 0.471 ± 0.220 | 0.523 ± 0.147 |
| DRU-Binary-B2 (quantum) | 0.568 ± 0.215 | 0.510 ± 0.162 | 0.510 ± 0.203 |

### ROC AUC

| Model | full | loose | strict |
|---|---|---|---|
| LogReg (classical) | 0.655 ± 0.300 | 0.660 ± 0.299 | 0.551 ± 0.146 |
| MLP (classical) | 0.693 ± 0.288 | 0.679 ± 0.281 | 0.619 ± 0.305 |
| RandomForest (classical) | 0.787 ± 0.184 | 0.773 ± 0.148 | 0.731 ± 0.114 |
| SVM-RBF (classical) | 0.670 ± 0.200 | 0.683 ± 0.186 | 0.581 ± 0.195 |
| XGBoost (classical) | 0.669 ± 0.162 | 0.704 ± 0.167 | 0.669 ± 0.151 |
| XGB-DRU-trained (hybrid) | 0.751 ± 0.223 | 0.542 ± 0.267 | 0.613 ± 0.169 |
| XGB-DRU-untrained (hybrid) | 0.533 ± 0.220 | 0.493 ± 0.212 | 0.566 ± 0.226 |
| XGB-PCA (hybrid) | 0.616 ± 0.213 | 0.657 ± 0.221 | 0.603 ± 0.226 |
| XGB-Poly2 (hybrid) | 0.626 ± 0.198 | 0.519 ± 0.169 | 0.673 ± 0.157 |
| XGB-RandomRBF (hybrid) | 0.801 ± 0.158 | 0.689 ± 0.147 | 0.564 ± 0.221 |
| XGB-raw (hybrid) | 0.453 ± 0.191 | 0.487 ± 0.208 | 0.558 ± 0.221 |
| DRU-Binary-B2 (quantum) | 0.757 ± 0.260 | 0.575 ± 0.313 | 0.557 ± 0.275 |

### FAR (normal)

| Model | full | loose | strict |
|---|---|---|---|
| LogReg (classical) | 0.433 ± 0.294 | 0.440 ± 0.296 | 0.520 ± 0.320 |
| MLP (classical) | 0.405 ± 0.344 | 0.420 ± 0.334 | 0.507 ± 0.361 |
| RandomForest (classical) | 0.454 ± 0.361 | 0.447 ± 0.362 | 0.480 ± 0.362 |
| SVM-RBF (classical) | 0.520 ± 0.333 | 0.519 ± 0.330 | 0.553 ± 0.407 |
| XGBoost (classical) | 0.455 ± 0.360 | 0.444 ± 0.361 | 0.513 ± 0.336 |
| XGB-DRU-trained (hybrid) | 0.504 ± 0.342 | 0.545 ± 0.285 | 0.451 ± 0.271 |
| XGB-DRU-untrained (hybrid) | 0.617 ± 0.449 | 0.515 ± 0.374 | 0.487 ± 0.350 |
| XGB-PCA (hybrid) | 0.619 ± 0.450 | 0.534 ± 0.421 | 0.455 ± 0.387 |
| XGB-Poly2 (hybrid) | 0.535 ± 0.423 | 0.448 ± 0.348 | 0.522 ± 0.327 |
| XGB-RandomRBF (hybrid) | 0.570 ± 0.390 | 0.627 ± 0.374 | 0.510 ± 0.348 |
| XGB-raw (hybrid) | 0.597 ± 0.443 | 0.541 ± 0.405 | 0.530 ± 0.324 |
| DRU-Binary-B2 (quantum) | 0.380 ± 0.358 | 0.533 ± 0.252 | 0.509 ± 0.187 |

## 2. DRU vs. best classical (per mode)

| Mode | Best classical (F1) | DRU (F1) | gap |
|---|---|---|---|
| full | LogReg = 0.650 | 0.568 | -0.082 |
| no_proxy_loose | MLP = 0.651 | 0.510 | -0.141 |
| no_proxy_strict | RandomForest = 0.545 | 0.510 | -0.035 |

## 3. XGB-DRU-trained vs. paired controls (F1 macro)

| Mode | DRU-trained | Poly2 | RandomRBF | DRU-untrained | beats all 3? |
|---|---|---|---|---|---|
| full | 0.509 | 0.404 | 0.498 | 0.345 | ✓ |
| no_proxy_loose | 0.529 | 0.496 | 0.444 | 0.471 | ✓ |
| no_proxy_strict | 0.561 | 0.531 | 0.509 | 0.540 | ✓ |

## 4. Degradation slope (full → strict, F1 macro)

| Model | full | strict | Δ |
|---|---|---|---|
| LogReg (classical) | 0.650 | 0.506 | -0.144 |
| MLP (classical) | 0.641 | 0.528 | -0.113 |
| RandomForest (classical) | 0.564 | 0.545 | -0.019 |
| SVM-RBF (classical) | 0.577 | 0.462 | -0.115 |
| XGBoost (classical) | 0.521 | 0.524 | +0.003 |
| XGB-DRU-trained (hybrid) | 0.509 | 0.561 | +0.051 |
| XGB-DRU-untrained (hybrid) | 0.345 | 0.540 | +0.195 |
| XGB-PCA (hybrid) | 0.368 | 0.532 | +0.163 |
| XGB-Poly2 (hybrid) | 0.404 | 0.531 | +0.127 |
| XGB-RandomRBF (hybrid) | 0.498 | 0.509 | +0.011 |
| XGB-raw (hybrid) | 0.345 | 0.523 | +0.178 |
| DRU-Binary-B2 (quantum) | 0.568 | 0.510 | -0.059 |

## 5. Conclusion

- **DRU collapses toward chance under `strict`** (F1 = 0.510); much of its observed performance under richer feature sets depends on contextual proxies.
- **Hybrid XGB-DRU-trained beats all three controls (Poly2, RandomRBF, DRU-untrained) under `strict`** by +0.021 F1 — evidence the learned quantum representation carries information not captured by deterministic or random non-linear expansions.

---
Generated automatically by §14 of TLM_DRU_FINAL.ipynb.