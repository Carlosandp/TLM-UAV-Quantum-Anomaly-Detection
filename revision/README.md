# Revision analyses

Analyses added for the revision of the manuscript. They reuse the data
construction, block assignment, and preprocessing of
`notebooks/TLM_DRU_FINAL.ipynb` without modification (the scripts execute the
notebook's own cells) and never overwrite the stored results in `results/`.

## Run

```bash
export TLM_DATASET_DIR=/path/to/dataset      # folder with ATT/, BARO/, BAT/, ...
cd revision/scripts
python rev_classical.py          # A0–A6
python rev_events.py             # A7 and label runs
python rev_dru.py d1             # D1: one stored circuit fit repeated, cost, latency
python rev_dru.py d2             # D2: circuit and hybrids without synthetic rows
python rev_figures.py            # revision/figures/rev_*.pdf
python rev_supplement_tables.py  # revision/supplement_tables/*.tex
```

Versions, run times, and exit codes of the last complete run:
[`results/run_log.txt`](results/run_log.txt).

## Outputs (`results/`)

| File(s) | Content |
|---|---|
| `A0_classical_blocked_smotetomek_*` | Stored classical results recomputed, with the false-alarm rate (`far`) and miss rate (`miss`) under their standard definitions |
| `A1_episodes.csv`, `A1_blocks.csv`, `A1_splits.csv`, `A1_label_runs.csv`, `A1_autocorr_strict.csv` | Structure of the log: gap-delimited segments, the ten time blocks, the ten partitions (future data in training, adjacency), the four fault intervals, autocorrelation of the physical variables |
| `A2_random_row_*` | Same models and preprocessing with a random stratified split of rows |
| `A3a_embargo_*` | Blocked split after removing training rows within 5, 15, 30, or 48 s of any test row |
| `A3b_forward_chaining_*` | Training on the past only (blocks `0..k-1` → block `k`) |
| `A3c_episode_holdout.csv` | Holding out each gap-delimited segment (not informative: single-class segments) |
| `A4_no_resampling_*`, `A4_smotetomek_vs_none.csv` | Classical models without SMOTETomek |
| `A4b_synthetic_diagnostics.csv` | Share of synthetic rows and simple plausibility checks |
| `A5_all_models_*`, `A5_paired_wins_strict.csv` | All twelve models with both error rates, derived from the stored per-run files; partition-by-partition comparison |
| `A6_*` | Why classical XGBoost and the five-variable XGBoost control differ; two label-informed classical controls |
| `A7_*` | Detection rate by whether the fault interval of a test sample also contributed rows to training |
| `D1_dru_cost_and_reproduction.json`, `D1_loss_history_seed0_strict.csv` | Reproduction of one stored circuit fit; circuit-evaluation count; wall-clock time; inference latency in simulation |
| `D2_strict_no_resampling_dru_hybrid_*` | Circuit, six hybrids, and two added controls trained on real rows only (physical-only inputs) |
| `revision_classical_log.txt`, `run_log.txt` | Console log and run record |

## Reading the results

- The ten partitions reshuffle ten time blocks of **one** log with **four**
  fault intervals. Standard deviations measure sensitivity to the partition,
  not variability across flights.
- Timings and latencies are measurements of a classical state-vector
  simulation. They do not describe quantum hardware.
- `D2` fit times were measured with five fits running in parallel and are not
  comparable with the serial time in `D1`.
