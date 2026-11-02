# Experimental Results and Artifacts

This directory contains all experimental outputs, prediction datasets, model performance logs, tables, and visualization figures produced by the benchmark and MTTD deployment scripts.

---

## 1. Inventory of Results Files

All results in this directory were generated against the canonical dataset (`data/cricket_hourly_workload.csv`, SHA-256: `0411f7b72cab109e31d0eed31da0c385c831ff85796dea49905b8b0926140c45`).

| File | Description | Generating Command | Dataset SHA-256 | Date Generated |
|---|---|---|---|---|
| `predictions_retrain.csv` | Full MTTD deployment hourly predictions (1,080 rows) with two-strike adaptive retraining | `python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_retrain.csv` | `0411f7b7...0c45` | 2026-11-02 |
| `predictions_noretrain.csv` | MTTD deployment hourly predictions (1,080 rows) with static active model (no retraining) | `python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_noretrain.csv --no-retrain` | `0411f7b7...0c45` | 2026-11-02 |
| `predictions_retrain_model_scores.csv` | Per-cycle evaluation metrics for all 5 candidate models across 45 deployment cycles (with retrain) | `python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_retrain.csv` | `0411f7b7...0c45` | 2026-11-02 |
| `predictions_retrain_model_summary.csv` | Aggregate performance summary for all 5 candidate models across 45 deployment cycles (with retrain) | `python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_retrain.csv` | `0411f7b7...0c45` | 2026-11-02 |
| `predictions_retrain_model_comparison.png` | Visualization of actual workload vs. predictions, active model timeline, and cycle scores | `python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_retrain.csv` | `0411f7b7...0c45` | 2026-11-02 |
| `predictions_noretrain_model_scores.csv` | Per-cycle evaluation metrics for all 5 candidate models across 45 deployment cycles (without retrain) | `python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_noretrain.csv --no-retrain` | `0411f7b7...0c45` | 2026-11-02 |
| `predictions_noretrain_model_summary.csv` | Aggregate performance summary for all 5 candidate models across 45 deployment cycles (without retrain) | `python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_noretrain.csv --no-retrain` | `0411f7b7...0c45` | 2026-11-02 |
| `predictions_noretrain_model_comparison.png` | Visualization of actual workload vs. predictions without retraining | `python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_noretrain.csv --no-retrain` | `0411f7b7...0c45` | 2026-11-02 |
| `run_manifest.json` | Execution metadata: dataset hash, row count, constants, trained models, fallback notes, library versions, timestamp | `mttd_bridge.py` | `0411f7b7...0c45` | 2026-11-02 |
| `tables/grand_summary.csv` | Synthetic workload consolidated results across all 4 patterns and 5 models | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `tables/table_uniform.csv` | Synthetic benchmark metrics on uniform pattern | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `tables/table_linear_low.csv` | Synthetic benchmark metrics on low linear growth pattern | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `tables/table_linear_high.csv` | Synthetic benchmark metrics on high linear growth pattern | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `tables/table_random.csv` | Synthetic benchmark metrics on random pattern | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `figures/fig1_workload_patterns.png` | Plot of all 4 synthetic workload patterns (paper Fig. 1) | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `figures/predictions_uniform.png` | Per-model prediction curves on uniform pattern | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `figures/predictions_linear_low.png` | Per-model prediction curves on low linear pattern | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `figures/predictions_linear_high.png` | Per-model prediction curves on high linear pattern | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `figures/predictions_random.png` | Per-model prediction curves on random pattern | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `figures/summary_comparison.png` | Bar charts comparing RMSE, MAE, and Score across patterns | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `figures/workload_patterns_preview.png` | Preview image showing synthetic pattern samples | Standalone script | N/A (synthetic) | 2026-11-02 |
| `logs/benchmark_all.log` | Console output log from full benchmark run (5 models) | `python run_benchmark.py` | N/A (synthetic) | 2026-11-02 |
| `logs/benchmark_original_only.log` | Console output log from paper-only benchmark run (3 models) | `python run_benchmark.py --original-only` | N/A (synthetic) | 2026-11-02 |
| `logs/mttd_bridge_retrain.log` | Console output log from MTTD run with adaptive retraining | `python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_retrain.csv` | `0411f7b7...0c45` | 2026-11-02 |
| `logs/mttd_bridge_noretrain.log` | Console output log from MTTD run without retraining | `python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_noretrain.csv --no-retrain` | `0411f7b7...0c45` | 2026-11-02 |

### Obsolete Results

No obsolete results remain in this repository. All earlier CSV files produced from previous uncanonical data versions were purged prior to regenerating this complete results set.

---

## 2. Schema of `*_model_scores.csv`

The `predictions_retrain_model_scores.csv` and `predictions_noretrain_model_scores.csv` files log candidate model performance per 24-hour deployment cycle.

| Column | Type | Description |
|---|---|---|
| `phase` | `str` | Deployment cycle label (`Deploy1` to `Deploy45`) |
| `start_hour` | `int` | Trace hour index where the cycle begins (e.g. 264, 288, ..., 1320) |
| `model` | `str` | Candidate model name (`ARIMA`, `SVR`, `RandomForest`, `XGBoost`, `NeuralNet-MLP`) |
| `RMSE` | `float` | Root Mean Squared Error computed over the 24 hours of this cycle |
| `MAE` | `float` | Mean Absolute Error computed over the 24 hours of this cycle |
| `Score` | `float` | Performance score percentage (Eq. 4) for this cycle |
| `is_active` | `int` | Binary flag: `1` if this model was the active forecasting model for this cycle, `0` otherwise |
| `threshold` | `float` | Adaptive threshold applied to this cycle (Eq. 5 or Eq. 6) |
| `retrain` | `int` | Binary flag: `1` if a 2-strike retraining event was triggered at this cycle boundary, `0` otherwise |

---

## 3. Schema of `*_model_summary.csv`

The `predictions_retrain_model_summary.csv` and `predictions_noretrain_model_summary.csv` files aggregate candidate model metrics across the entire 45-cycle deployment window (1,080 hours).

| Column | Type | Description |
|---|---|---|
| `model` | `str` | Candidate model name |
| `mean_Score` | `float` | Arithmetic mean of the model's 45 individual cycle scores (%) |
| `mean_RMSE` | `float` | Arithmetic mean of the model's 45 individual cycle RMSE values |
| `mean_MAE` | `float` | Arithmetic mean of the model's 45 individual cycle MAE values |
| `cycles` | `int` | Total deployment cycles evaluated (45) |
| `cycles_best` | `int` | Number of cycles in which this candidate had the strictly lowest Score among all 5 models |
| `overall_Score` | `float` | Performance score percentage (Eq. 4) evaluated across all 1,080 deployment hours continuously |
| `hours_active` | `int` | Total number of hours this model was selected and actively deployed to serve forecasts |
