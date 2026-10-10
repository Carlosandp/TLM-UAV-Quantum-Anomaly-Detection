# Data — TLM:UAV

The full data set is **not** redistributed in this repository.

## 1. Source

> Yang, T., Lu, Y., Deng, H., Chen, J., & Tang, X. (2023).
> *Acquisition and Processing of UAV Fault Data Based on Time Line Modeling
> Method.* **Applied Sciences**, 13(7), 4301.
> https://doi.org/10.3390/app13074301

Obtain the data from the authors' public release and keep its folder layout.

## 2. What the pipeline reads

The notebook and the revision scripts read the **raw per-sensor files**:

```
<TLM_DATASET_DIR>/
├── ATT/ALL_FAIL_LOG_ATT.csv
├── BARO/ALL_FAIL_LOG_BARO.csv
├── BAT/ALL_FAIL_LOG_BAT_0.csv
├── CTUN/ALL_FAIL_LOG_CTUN.csv
├── GPS/ALL_FAIL_LOG_GPS_0.csv
├── IMU/ALL_FAIL_LOG_IMU_0_Random.csv
├── MAG/ALL_FAIL_LOG_MAG_0.csv
├── MOTB/ALL_FAIL_LOG_MOTB.csv
├── PSCD/ALL_FAIL_LOG_PSCD.csv
├── RATE/ALL_FAIL_LOG_RATE.csv
├── VIBE/ALL_FAIL_LOG_VIBE_0_Random.csv
└── XKF1/ALL_FAIL_LOG_XKF1_0_Random.csv
```

`TLM_DATASET_DIR` defaults to `data/dataset/` under the repository root. The
merged working table (4,817 rows, 72 numeric variables after dropping
constant columns) is rebuilt from these files on every run.

Labels: `0` nominal; `1` GPS fault; `2` accelerometer fault; `3` engine
fault; `4` remote-control fault. The mapping was checked against the signals
(see `docs/methodology.md`). The faults are simulated; the data contain no
attacks.

## 3. `Fusion_Data.csv` is not the experimental source

The release also contains a ready-made merged table, `Fusion_Data.csv`
(12,253 rows, 17 features). It is **not used** for any result in this
repository. An audit found two pairs of exactly duplicated columns in it
(`ErrRP`/`ErrYaw` and `MagY`/`MagZ`) that are not duplicates in the raw
sensor files.

[`sample/Fusion_Data_sample.csv`](sample/Fusion_Data_sample.csv) holds the
first 100 rows of that table as a format reference only. SHA-256 of the full
file that was audited:

```
e17dc97292bbf32f96424902a78c64d23b8185fe6002d5f008a4d7f813244465  Fusion_Data.csv
```
