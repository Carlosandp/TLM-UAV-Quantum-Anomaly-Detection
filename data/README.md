# Dataset — TLM:UAV (Fusion_Data.csv)

This repository does **not** redistribute the full `Fusion_Data.csv` file.
A 100-row sample is provided in [`sample/Fusion_Data_sample.csv`](sample/Fusion_Data_sample.csv)
for smoke-testing the pipeline (column schema, types, label encoding).

## 1. Source

The dataset is derived from the publicly released TLM:UAV software-in-the-loop
telemetry corpus:

> Yang, T., Lu, Y., Deng, H., Chen, J., & Tang, X. (2023).
> *Acquisition and Processing of UAV Fault Data Based on Time Line Modeling
> Method.* **Applied Sciences**, 13(7), 4301.
> https://doi.org/10.3390/app13074301

The article is published Open Access under **CC BY 4.0**, which permits
redistribution with attribution. The 100-row sample committed here is
provided under the same CC BY 4.0 terms inherited from the source.

## 2. How to obtain the full dataset

The full processed file used in this study is `Fusion_Data.csv`
(12,253 rows + 1 header line). Obtain it via one of the following routes:

1. **Authors' release.** Follow the data-availability statement of the
   TLM:UAV article (DOI above) and download the supplementary archive.
2. **Local preprocessing.** Concatenate and align the per-sensor CSVs
   (`ATT`, `BARO`, `BAT`, `CTUN`, `GPS`, `IMU`, `MAG`, `MOTB`, `PSCD`,
   `RATE`, `VIBE`, `XKF1`) on the `timestamp` column following the
   pipeline described in the article. The output schema must match the
   columns listed below.

Place the resulting file at:

```
data/dataset/Fusion_Data.csv
```

(this path is the default expected by the notebook when `TLM_UAV_ROOT`
is the repository root; see `docs/reproducibility.md`).

## 3. Expected schema

19 columns, all numeric. Header (exact order):

```
timestamp, DesRoll, Roll, DesPitch, Pitch, DesYaw, Yaw, ErrRP, ErrYaw,
MagX, MagY, MagZ, abGyrX, abGyrY, abGyrZ, abAccX, abAccY, abAccZ, labels
```

- `timestamp` — microseconds since boot (used for group-aware temporal
  splitting under the B2 protocol; **never used as a feature**).
- 17 numeric telemetry features (attitude setpoints, attitude estimates,
  magnetometer, gyro bias, accelerometer bias).
- `labels` — integer in `{0, 1, 2, 3, 4}`:
  - `0` — Normal (≈ 53.6 % of rows)
  - `1`–`4` — Distinct fault-injection classes (imbalance ratio ≈ 11.9×)

For the binary task used in the main benchmark, labels `1`–`4` are
collapsed into a single `anomaly` class.

## 4. Integrity check

The exact file used in this study has the following SHA-256:

```
e17dc97292bbf32f96424902a78c64d23b8185fe6002d5f008a4d7f813244465  Fusion_Data.csv
```

Verify with:

```bash
shasum -a 256 data/dataset/Fusion_Data.csv
```

If your hash differs, the preprocessing or alignment may not match the
one used in the paper; results may not be numerically reproducible (but
the pipeline will still run).

## 5. Sample file

[`sample/Fusion_Data_sample.csv`](sample/Fusion_Data_sample.csv) contains
the first 100 data rows of `Fusion_Data.csv` (all label `0`, since
faults appear later in the temporal sequence). It is intended only as a
schema/format reference and for smoke-testing, **not** for training or
evaluation.
