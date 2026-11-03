# Results

All cricket results were produced from `data/cricket_hourly_workload.csv` (SHA-256 `0411f7b72cab109e31d0eed31da0c385c831ff85796dea49905b8b0926140c45`).

## `cricket/` — MTTD on the 2023 Cricket World Cup trace

| Files | Produced by |
|---|---|
| `predictions_all5.csv`, `predictions_all5_model_scores.csv`, `predictions_all5_model_summary.csv`, `predictions_all5_model_comparison.png` | `python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/cricket/predictions_all5.csv` (standard MTTD) |
| `predictions_daily.csv`, `predictions_daily_model_*` | same, with `--reselect-every 24` (daily reselection) |
| `predictions_hourly.csv`, `predictions_hourly_model_*` | same, with `--reselect-every 1` (hourly reselection) |
| `cloudsim/console_*.jpeg` | Screenshots of the three CloudSim 3.0.3 runs (`cloudsim/ProvisioningSimulation.java`) |
| `cloudsim/cloudsim_all5.csv`, `cloudsim_daily.csv`, `cloudsim_hourly.csv` | Per-hour CloudSim output of the three runs |

### File formats

- `predictions_*.csv`: `hour, actual, predicted, active_model, retrain_event` — one row per deployment hour (264–1343). `active_model` is the model whose forecast was used in that hour; `retrain_event = 1` marks the last hour of a cycle that triggered retraining.
- `*_model_scores.csv`: one row per model per phase (`Test`, `Deploy1`…`Deploy45`, `Retest@<hour>`) with `RMSE`, `MAE`, `Score`, `is_active`, `threshold`, `retrain`.
- `*_model_summary.csv`: per model over all 45 cycles — `mean_Score`, `mean_RMSE`, `mean_MAE`, `cycles`, `cycles_best` (cycles with the lowest Score), `overall_Score` (Score over all 1,080 hours), `hours_active`.
- `cloudsim/cloudsim_*.csv`: `hour, actual, forecast, vms_provisioned, vms_needed, completion_s, sla_violated`.

### Summary

| Run | Score % | RMSE | MAE | Switches | Retrains | CloudSim VM-hours | SLA-violating hours |
|---|---|---|---|---|---|---|---|
| Standard MTTD | 32.78 | 13,649.96 | 5,945.88 | 0 | 1 (hour 552) | 7,475 | 145 |
| Daily reselection | 21.41 | 8,412.83 | 4,239.58 | 27 | 0 | 6,802 | 263 |
| Hourly reselection | 18.96 | 7,757.50 | 3,536.08 | 77 | 1 (hour 864) | 7,312 | 158 |

## `tables/`, `figures/`, `logs/` — synthetic benchmark

Produced by `python run_benchmark.py` (all five models) and `python run_benchmark.py --original-only` (ARIMA, SVR, Random Forest) on the four synthetic patterns (697 minutes, 70/30 split, 100-point test window, window 5): per-pattern tables, `grand_summary.csv`, prediction plots, `summary_comparison.png`, `fig1_workload_patterns.png` and the console logs.
