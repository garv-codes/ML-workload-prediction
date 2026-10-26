# Machine Learning Workload Prediction in Cloud Environments

**Dynamic Multi-Model Framework with Adaptive Retraining (MTTD)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports%202026-blue.svg)](https://www.nature.com/articles/s41598-026-40777-z)

---

## 1. Overview

Cloud platforms rely on proactive autoscaling to maintain SLAs while minimising
resource over-provisioning.  Reactive, threshold-based autoscaling responds
**after** traffic spikes, causing latency penalties and wasted capacity.

This repository re-implements the **MTTD (Monitor–Train–Test–Deploy)** framework
from [Nashaat et al., *Sci. Rep.* 16, 10983 (2026)](https://www.nature.com/articles/s41598-026-40777-z)
and extends it with additional ML/DL architectures and a semi-synthetic cricket
workload dataset.

### Objectives

| # | Objective | Description |
|---|-----------|-------------|
| 1 | **Paper Reproduction** | Implement all 4 MTTD phases with ARIMA, SVR, and Random Forest using the paper's exact mathematical formulations. |
| 2 | **Novel Extensions** | Introduce **XGBoost** (gradient boosting vs. bagging) and **LSTM** (deep recurrent modelling) as additional candidate models, along with **MAPE** as a scale-invariant metric. |
| 3 | **Semi-Synthetic Validation** | Evaluate the framework on a **Cricket Hourly Workload** trace built from real Wikimedia daily page views with a synthetic hourly profile (see [§4 Dataset](#4-dataset)). |
| 4 | **Deployment API** | Provide a clean `predictors.py` streaming interface for integration with autoscaling controllers. |

---

## 2. Framework Architecture

The MTTD framework operates across four continuous phases:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 1 — Workload Identification & Preprocessing                      │
│    • Synthetic traces: Uniform, Linear (Low/High), Random               │
│    • Semi-synthetic trace: Cricket Hourly Workload                      │
│    • Sliding window feature extraction (window = 5 timesteps)           │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 2 — Candidate Model Training                                     │
│    • Baseline: ARIMA(p,d,q), SVR (RBF kernel), Random Forest            │
│    • Novel: XGBoost Regressor, LSTM (PyTorch / TensorFlow / MLP)        │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 3 — Testing & Dynamic Model Selection                            │
│    • Compute RMSE, MAE, MAPE                                            │
│    • Performance Score = √(RMSE² + MAE²)                                │
│    • Active Model = argmin(Score)                                        │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 4 — Continuous Deployment & Adaptive Retraining                  │
│    • Deploy active model for one-step-ahead forecasting                 │
│    • Strike 1: Score > Threshold → tighten threshold (λ₂)              │
│    • Strike 2: consecutive breach → trigger full retraining             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deviations from the Paper

| Aspect | Paper (Nashaat et al. 2026) | This Repository |
|--------|----------------------------|-----------------|
| **Dataset** | FIFA World Cup 1998 / 2018 server request logs | Wikimedia daily page views for the 2023 Cricket World Cup, expanded to hourly with a synthetic profile.  Results are **not directly comparable** to the paper's Tables 10-13. |
| **Ensemble Models** | Random Forest (bagging) | Random Forest + **XGBoost** (gradient boosting) |
| **Deep Learning** | Not evaluated | **LSTM** with PyTorch, TensorFlow, and MLP fallback |
| **Error Metrics** | RMSE, MAE, Score | RMSE, MAE, **MAPE**, Score |
| **Deployment API** | Monolithic evaluation | Modular `predictors.py` streaming interface |

> **Note:** The paper's own FIFA datasets are available at
> <https://github.com/walidmoussa2010/Workload-prediction->.

---

## 4. Dataset

### 4.1 Synthetic Workload Traces

Four synthetic patterns reproduce the scenarios from Fig. 1 of the original paper:

| Pattern | Description | Generator |
|---------|-------------|-----------|
| Uniform (Bell) | Ramps to peak then descends | `generate_uniform_workload()` |
| Linear Low | Steady growth at 0.35 req/s/min | `generate_linear_workload(rate=0.35)` |
| Linear High | Steep growth at 2.31 req/s/min | `generate_linear_workload(rate=2.31)` |
| Random | High volatility, bounded random walk | `generate_random_workload()` |

### 4.2 Cricket Hourly Workload (Semi-Synthetic)

A **semi-synthetic** workload trace built from real Wikimedia daily page-view
counts for the 2023 Cricket World Cup.  See [`data/README.md`](data/README.md)
for full provenance.

#### What is real / what is synthetic

| Aspect | Status | Details |
|--------|--------|---------|
| Daily page-view totals | **Real** (measured) | 57 daily counts from the Wikimedia Pageviews API for `en.wikipedia`, article `2023_Cricket_World_Cup`, Oct 5 – Nov 30 2023. |
| Hourly 70/30 peak/off-peak split | **Synthetic** | 70 % of each day's views assigned to hours 14:00–21:00 (IST match hours); 30 % to remaining hours. |
| Within-day shape | **Synthetic** | Raised-cosine profile centred on 18:00 for peak hours; uniform for off-peak. |
| Flash-crowd noise and spikes | **Synthetic** | Multiplicative log-normal noise (σ = 0.15) injected per hour, then renormalised. |
| Daily total preservation | **Exact** | After noise, hourly values are renormalised so each day's sum equals the original daily total (max diff ≤ 1). |
| Seed | Fixed (42) | Deterministic; same seed → identical file. |

#### Why this dataset?

The paper's FIFA server logs (1998/2018) are old and — for the 1998 trace —
sparse at hourly granularity.  A 57-point daily series is too short to train
time-series models on, hence the hourly expansion.  This expansion adds
resolution that is **modelled, not measured**; it does not "solve" sparsity in
the sense of recovering true hourly patterns.  Wikipedia page views are also a
**proxy** for user traffic, not server request logs.

#### How to regenerate the data

```bash
python data/build_hourly_workload.py --seed 42
```

The script reads `data/raw/wikimedia_daily.csv` (cached offline).  Pass
`--refresh` to re-fetch from the Wikimedia API.  Running with the same seed
always produces a byte-identical output file.

---

## 5. Project Structure

```
ML-workload-prediction/
├── README.md                    # This file
├── requirements.txt             # Pinned dependencies
├── .gitignore
│
├── data/
│   ├── README.md                # Dataset provenance & documentation
│   ├── build_hourly_workload.py # Generates the semi-synthetic hourly CSV
│   ├── cricket_hourly_workload.csv  # Canonical 1,344-row hourly trace
│   └── raw/
│       └── wikimedia_daily.csv  # Cached 57-row daily page-view data
│
├── synthetic_workload.py        # Phase 1: Synthetic workload generators (4 patterns)
├── sliding_window.py            # Phase 1: Sliding window time-series transform
│
├── arima_baseline.py            # Phase 2: ARIMA auto-fit, metrics (RMSE, MAE, MAPE, Score)
├── svr_rf_models.py             # Phase 2: SVR (RBF) & Random Forest with GridSearchCV
├── xgboost_model.py             # Phase 2: XGBoost regressor (novel)
├── lstm_model.py                # Phase 2: LSTM with multi-backend support (novel)
│
├── model_selector.py            # Phase 3: Model evaluation & active model selection
├── deployment_manager.py        # Phase 4: Two-tier adaptive threshold & retraining
├── predictors.py                # Streaming inference wrappers for deployment
│
├── run_benchmark.py             # End-to-end benchmark (Tables 5–8 + novel comparisons)
├── workload_patterns_preview.png
└── results/                     # Auto-generated outputs
    ├── figures/                  #   Prediction plots, workload patterns, summary charts
    └── tables/                   #   Per-pattern CSVs and grand summary
```

---

## 6. Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/garv-codes/ML-workload-prediction.git
cd ML-workload-prediction

# Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** For PyTorch on Windows without a GPU, install the CPU-only wheel:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 7. Usage

### Full Benchmark (All 5 Models)

Runs ARIMA, SVR, Random Forest, XGBoost, and LSTM across all synthetic
workload patterns.  Generates comparison tables, prediction plots, and
summary charts in `results/`.

```bash
python run_benchmark.py
```

### Paper-Only Reproduction (3 Models)

Runs only the original ARIMA, SVR, and Random Forest models to reproduce
Tables 5–8.

```bash
python run_benchmark.py --original-only
```

### Individual Module Testing

Each module is self-contained and can be executed independently:

```bash
python synthetic_workload.py      # Generate and preview workload patterns
python arima_baseline.py          # ARIMA baseline evaluation
python svr_rf_models.py           # SVR and Random Forest evaluation
python xgboost_model.py           # XGBoost evaluation
python lstm_model.py              # LSTM evaluation
python model_selector.py          # Phase 3: model selection across all patterns
python deployment_manager.py      # Phase 4: adaptive threshold simulation
```

---

## 8. Mathematical Formulation

### Error Metrics

**Root Mean Squared Error (RMSE):**

$$RMSE = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(y_t - \hat{y}_t)^2}$$

**Mean Absolute Error (MAE):**

$$MAE = \frac{1}{N}\sum_{t=1}^{N}|y_t - \hat{y}_t|$$

**Mean Absolute Percentage Error (MAPE) — Novel:**

$$MAPE = \frac{1}{N}\sum_{t=1}^{N}\left|\frac{y_t - \hat{y}_t}{y_t}\right| \times 100\%$$

### Performance Score

$$\text{Score}_i = \sqrt{RMSE_i^2 + MAE_i^2}$$

$$\text{Active Model} = \arg\min_i (\text{Score}_i)$$

### Adaptive Retraining Thresholds

| Cycle | Formula | Purpose |
|-------|---------|---------| 
| $t+1$ | $\text{Threshold}_{t+1} = \text{Score}_t + \lambda_1 \times \text{Score}_t \quad (\lambda_1 = 0.75)$ | Initial tolerance band |
| $t+2$ | $\text{Threshold}_{t+2} = \text{Score}_{t+1} + \lambda_2 \times \text{Score}_{t+1} \quad (\lambda_2 = 0.50)$ | Tightened after first breach |

**Retraining is triggered** when the score exceeds the threshold on two
consecutive deployment cycles.

---

## 9. Results

### 9.1 Synthetic Workload Patterns

Results from `run_benchmark.py` across the four synthetic patterns
(697 time steps, 70/30 train-test split, 100-point test window, window=5):

| Pattern | Model | RMSE | MAE | MAPE | Score (%) | Best? |
|---------|-------|------|-----|------|-----------|-------|
| uniform | ARIMA | 2.2869 | 1.8859 | 1.42% | 1.51 | ✓ |
| uniform | SVR | 16.8411 | 13.9486 | 11.49% | 11.11 | |
| uniform | RandomForest | 5.7550 | 5.1867 | 3.96% | 3.95 | |
| uniform | XGBoost | 6.3592 | 5.6850 | 4.31% | 4.35 | |
| uniform | LSTM | 5.4387 | 4.8183 | 3.80% | 3.70 | |
| linear_low | ARIMA | 1.0289 | 0.8201 | 0.44% | 0.49 | ✓ |
| linear_low | SVR | 97.2096 | 94.1927 | 49.73% | 50.95 | |
| linear_low | RandomForest | 21.1237 | 18.5355 | 9.61% | 10.56 | |
| linear_low | XGBoost | 20.9576 | 18.3463 | 9.51% | 10.46 | |
| linear_low | LSTM | 4.4614 | 4.1254 | 2.17% | 2.29 | |
| linear_high | ARIMA | 1.0375 | 0.8259 | 0.07% | 0.08 | ✓ |
| linear_high | SVR | 674.8332 | 671.4941 | 54.05% | 54.28 | |
| linear_high | RandomForest | 135.9055 | 118.4355 | 9.29% | 10.25 | |
| linear_high | XGBoost | 136.2235 | 118.8002 | 9.32% | 10.28 | |
| linear_high | LSTM | 27.1023 | 25.8980 | 2.06% | 2.14 | |
| random | ARIMA | 8.2045 | 6.5380 | 3.28% | 3.63 | ✓ |
| random | SVR | 18.6115 | 12.8222 | 7.00% | 7.73 | |
| random | RandomForest | 8.8420 | 7.2290 | 3.66% | 3.96 | |
| random | XGBoost | 8.5104 | 7.0343 | 3.55% | 3.83 | |
| random | LSTM | 8.2452 | 6.7314 | 3.40% | 3.69 | |

### 9.2 Semi-Synthetic Cricket Trace (real daily totals, synthetic hourly profile)

*Results pending.*  The cricket hourly workload dataset
(`data/cricket_hourly_workload.csv`) is available for benchmarking but has not
yet been run through `run_benchmark.py`.

> **Caveat:** The daily seasonality (24-hour cycle, match-hour peak) in this
> dataset is **built into the data by construction** — models learning this
> pattern is expected and is not evidence about real traffic behaviour.

### Output Artifacts

All generated results are saved to the `results/` directory:

- `results/figures/fig1_workload_patterns.png` — Workload pattern visualisation
- `results/figures/predictions_<pattern>.png` — Per-pattern prediction plots
- `results/figures/summary_comparison.png` — Cross-model comparison chart
- `results/tables/table_<pattern>.csv` — Per-pattern metric tables
- `results/tables/grand_summary.csv` — Consolidated results across all patterns

---

## 10. Limitations

1. **Synthetic hourly profile:** The intra-day shape (70/30 peak/off-peak split,
   raised-cosine profile, log-normal noise) is entirely synthetic.  Models that
   learn the 24-hour cycle or match-hour peak are learning a pattern that was
   **injected by the build script**, not observed from real traffic.

2. **Flash-crowd spikes are noise:** The spikes in the hourly trace come from
   multiplicative log-normal noise, not from real flash-crowd events.  Model
   performance on these spikes does not validate real-world spike handling.

3. **Page views ≠ server requests:** Wikipedia page views are a proxy for
   user interest, not actual server request logs (which include API calls,
   static assets, bots, etc.).  The paper's FIFA dataset consists of HTTP
   request logs, which have different statistical properties.

4. **Results not comparable to paper:** Because the dataset differs (semi-synthetic
   Wikipedia page views vs. real FIFA server logs), the numbers in §9 cannot
   be directly compared to the paper's Tables 10-13.

---

## 11. Deployment Integration

The `predictors.py` module provides a unified streaming interface for
integrating trained models into autoscaling controllers:

```python
from deployment_manager import DeploymentManager
from predictors import make_rf_predictor

# Initialise with the active model's Phase 3 score
manager = DeploymentManager(active_model_name="RandomForest", initial_score=1.42)

# Create a predictor wrapping the fitted model
predictor = make_rf_predictor(fitted_model, window=5, initial_history=recent_data)

# Predict next minute's traffic
predicted = predictor.predict_next()

# Feed observed value back
predictor.observe(actual_value)

# Run deployment cycle and check for retraining
status = manager.run_cycle(actual_values=[...], predicted_values=[...])
if status["retrain_triggered"]:
    # Trigger Phase 2 & 3 reselection
    pass
```

### CloudSim Simulation

The `cloudsim/ProvisioningSimulation.java` class evaluates VM provisioning policies (Reactive, MTTD without retraining, MTTD with retraining, Oracle, and MTTD-Hybrid) using predictions generated by `mttd_bridge.py`.

> **Note on Published Numbers:** In `ProvisioningSimulation.java`, `MAX_HOURS = 100` is configured by default for rapid testing. Any published or reported results must come from a **full simulation run** (`MAX_HOURS = 0`, simulating all hours in the trace).

---

## 12. References

1. Nashaat, M., et al. "A dynamic machine learning approach for workload prediction in cloud environments." *Scientific Reports*, vol. 16, no. 1, 2026. DOI: [10.1038/s41598-026-40777-z](https://www.nature.com/articles/s41598-026-40777-z)
2. Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *ACM KDD*, 2016.
3. Hochreiter, S., & Schmidhuber, J. "Long Short-Term Memory." *Neural Computation*, 1997.

---

## 13. License

This project is licensed under the [MIT License](LICENSE).
