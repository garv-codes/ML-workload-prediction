# Machine Learning Workload Prediction in Cloud Environments

**Reproducing and Extending the MTTD Dynamic Model-Selection Framework: Five ML Models on a 2023 Cricket World Cup Workload with CloudSim Evaluation**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports%202026-blue.svg)](https://www.nature.com/articles/s41598-026-40777-z)
[![Research Paper](https://img.shields.io/badge/PDF-Research%20Paper-red.svg)](paper/MTTD_CloudSim_Cricket_WC2023_Research_Paper.pdf)

**Authors:** Abhinav Mishra, Garv Gujral, Shashank Yadav  
**Affiliation:** Netaji Subhas University of Technology, New Delhi 110075, India  
**Repository:** `https://github.com/garv-codes/ML-workload-prediction`

---

## 1. Overview

Cloud computing platforms rely on proactive autoscaling to maintain Service Level Agreements (SLAs) while minimising expensive resource over-provisioning. Traditional reactive autoscalers respond only after demand shifts occur, incurring latency penalties and SLA violations during rapid traffic bursts.

This repository reproduces and extends the **MTTD (Monitor–Train–Test–Deploy)** framework introduced by [Nashaat et al., *Scientific Reports* 16, 10983 (2026)](https://www.nature.com/articles/s41598-026-40777-z). Our study extends the original framework along four dimensions:
1. **Model Pool Expansion:** Evaluates five models (ARIMA, SVR, Random Forest, XGBoost, and PyTorch LSTM) rather than the original three candidate models.
2. **Real-World Tournament Workload:** Replaces the 1998/2018 FIFA server traces with a 56-day, 1,344-hour semi-synthetic workload trace based on Wikipedia page views from the 2023 ICC Men's Cricket World Cup.
3. **CloudSim Infrastructure Provisioning:** Translates predictive forecasts directly into datacenter provisioning decisions across five allocation policies (REACTIVE, MTTD-noRetrain, MTTD-Retrain, ORACLE, and MTTD-Hybrid) simulating 1,080 continuous hours in CloudSim 3.0.3.
4. **Reselection-Frequency Study:** Investigates model reselection dynamics under varying observation windows and frequencies, demonstrating that higher-frequency reselection substantially improves prediction accuracy and infrastructure efficiency.

---

## 2. Framework Architecture

The MTTD framework operates continuously through four coordinated phases, followed by cloud datacenter simulation:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 1 — Monitoring                                                    │
│    • Aggregate incoming traffic requests into discrete time series       │
│    • Sliding window features:                                            │
│        - Synthetic benchmark: WINDOW = 5 timesteps                       │
│        - Cricket workload: WINDOW = 24 hours                             │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 2 — Training                                                      │
│    • Fit candidate models on recent history:                             │
│        - Cricket workload: TRAIN_LEN = 240 hours (10 days)               │
│    • Model pool: ARIMA, SVR (RBF), Random Forest, XGBoost, LSTM (PyTorch)│
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 3 — Testing and Model Selection                                   │
│    • Evaluate candidate one-step-ahead forecasts across the test window: │
│        - Cricket workload: TEST_LEN = 24 hours (1 day)                   │
│    • Compute Eq. 4 Score = 100 * (0.5 * RMSE / RMS(y) + 0.5 * MAE / Mean(y))
│    • Deploy Active Model = argmin(Score)                                 │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 4 — Continuous Deployment & Adaptive Retraining                   │
│    • Active model issues one-step-ahead forecasts each timestep          │
│    • Cricket workload: CYCLE = 24 hours (45 deployment cycles = 1,080 h) │
│    • At each cycle boundary, compute cycle Score and evaluate threshold: │
│        - Healthy: Score <= Threshold -> reset strikes; refresh (Eq. 5)   │
│        - Strike 1: Score > Threshold -> tighten threshold (Eq. 6)        │
│        - Strike 2: Consecutive breach -> trigger Phase 2/3 retraining    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CloudSim Provisioning Simulation                                        │
│    • Map hourly workload forecasts to VM allocation decisions            │
│    • Simulate 1,080 deployment hours across 5 provisioning policies      │
│    • Measure VM-hours, wasted capacity, SLA violations, and makespan     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Provenance: Paper vs. Our Extensions

| Aspect | Paper (Nashaat et al., 2026) | This Repository |
|---|---|---|
| **Core Framework** | 4-phase MTTD pipeline (Monitor, Train, Test, Deploy) | Faithful reproduction of the 4-phase MTTD loop |
| **Model Selection & Thresholds** | Equations 2–6 as published | Eq. 2–6 implemented; adaptive 2-strike state machine |
| **Candidate Model Pool** | 3 candidate models (ARIMA, SVR, Random Forest) | 5 candidate models (ARIMA, SVR, Random Forest, XGBoost, PyTorch LSTM) |
| **Novel Model 1** | Not evaluated | **XGBoost Regressor** (gradient-boosted trees) |
| **Novel Model 2** | Evaluated separately in discussion | **PyTorch LSTM** recurrent neural network |
| **Dataset** | FIFA World Cup 1998 and 2018 HTTP server access logs | 2023 Cricket World Cup, daily Wikipedia page views (agent = user), 1 Oct – 25 Nov 2023, expanded to hourly |
| **Synthetic Benchmarks** | 4 synthetic patterns (uniform, linear low, linear high, random) | All 4 patterns evaluated across all 5 candidate models |
| **CloudSim Evaluation** | Conceptual formulation | **CloudSim provisioning evaluation (5 policies)** over 1,080 hours |
| **Reselection Study** | Fixed 2-strike cycle retraining only | **Reselection-frequency study** comparing 6 dynamic selection strategies |

---

## 4. How the MTTD Loop Runs (`mttd_bridge.py`)

The end-to-end continuous deployment simulation is implemented in `mttd_bridge.py`:

- **Operating Parameters:**
  - `WINDOW = 24` hours
  - `TRAIN_LEN = 240` hours (10 days)
  - `TEST_LEN = 24` hours (1 day)
  - `CYCLE = 24` hours (1 day)
  - `lambda1 = 0.75` (healthy tolerance band, Eq. 5)
  - `lambda2 = 0.50` (tightened tolerance band after Strike 1, Eq. 6)
- **Hourly Forecast:** The designated active model issues a one-step-ahead forecast for each hour.
- **Cycle Boundary Evaluation:** At the end of every 24-hour deployment cycle, the cycle Score is computed using Eq. 4.
- **Adaptive Threshold State Machine:**
  - **Healthy Cycle ($\text{Score} \le \text{Threshold}$):** Strikes reset to 0; threshold refreshes to $\text{Score} \times (1 + \lambda_1)$ (Eq. 5).
  - **Strike 1 ($\text{Score} > \text{Threshold}$):** First breach logged; threshold tightens to $\text{Score} \times (1 + \lambda_2)$ (Eq. 6); active model remains deployed.
  - **Strike 2 (Second Consecutive Breach):** **Retraining is triggered.**
- **Retraining Procedure:** When triggered at a cycle boundary, all 5 candidate models are retrained on the most recent 264 hours (`TRAIN_LEN + TEST_LEN`), re-tested on the trailing 24 hours (`TEST_LEN`), and the candidate with the lowest test Score becomes the new active model.
- **Stability Guarantee:** A single anomalous cycle never triggers retraining. Model switches occur exclusively at 24-hour cycle boundaries following two consecutive threshold breaches.

---

## 5. Dataset

### 5.1 Synthetic Workload Traces

Four synthetic benchmark patterns reproduce the workload scenarios in Fig. 1 of Nashaat et al. (2026):

| Pattern | Description | Generator Function |
|---|---|---|
| **Uniform** | Bell-shaped ramp to peak and descent | `generate_uniform_workload()` |
| **Linear Low** | Low-rate steady growth (0.35 req/s/min) | `generate_linear_workload(rate=0.35)` |
| **Linear High** | High-rate steep growth (2.31 req/s/min) | `generate_linear_workload(rate=2.31)` |
| **Random** | Volatile bounded random walk | `generate_random_workload()` |

### 5.2 2023 Cricket World Cup Hourly Workload

The real-world workload trace covers the 2023 ICC Men's Cricket World Cup held in India (56 days, 1 Oct – 25 Nov 2023). For complete provenance, export settings, and validation procedures, see [`data/README.md`](data/README.md).

| Property | Value |
|---|---|
| **Source** | Wikimedia Pageviews tool (`pageviews.wmcloud.org`), article "2023 Cricket World Cup", project `en.wikipedia` |
| **Agent Filter** | **user** only (automated traffic and bots excluded) |
| **Raw File** | [`data/raw/pageviews-2023-10-01-2023-11-25.csv`](data/raw/pageviews-2023-10-01-2023-11-25.csv) (56 daily totals) |
| **Hourly File** | [`data/cricket_hourly_workload.csv`](data/cricket_hourly_workload.csv) (1,344 hourly observations, `Timestamp,Hits`) |
| **Intra-Day Split** | 70% of daily total spread over match window 14:00–21:59 (8 hours, 8.75% each); 30% over remaining 16 hours (1.875% each) |
| **Noise Model** | Multiplied by uniform noise $\mathcal{U}(0.92, 1.08)$ (uniform $\pm 8\%$), rescaled to preserve daily sum $D$, integer-rounded (remainder to peak hour) |
| **Random Seed** | Fixed seed 42 (`numpy.random.default_rng`) for deterministic reproducibility |
| **Key Statistics** | Total page views: **29,957,762**; Peak hour: **117,612** at `2023-11-19 17:00:00` (hour index 1193, Final match) |
| **Theoretical Noise Floor** | **4.06%** (theoretical minimum Score under uniform $\pm 8\%$ noise) |

---

## 6. Project Structure

```
ML-workload-prediction/
├── README.md                                 # Root documentation and benchmark results
├── requirements.txt                          # Dependencies and environment version manifest
├── LICENSE                                   # MIT License
├── .gitattributes                            # Binary preservation for dataset files
├── .gitignore                                # Build and cache ignore patterns
│
├── data/
│   ├── README.md                             # Dataset provenance, schema, and validation
│   ├── cricket_hourly_workload.csv           # Canonical 1,344-row hourly cricket dataset
│   ├── build_hourly_workload.py              # Canonical daily-to-hourly workload builder
│   └── raw/
│       └── pageviews-2023-10-01-2023-11-25.csv # Raw 56-day Wikimedia daily page views
│
├── cloudsim/
│   ├── README.md                             # CloudSim compilation and execution guide
│   └── ProvisioningSimulation.java           # Java CloudSim datacenter provisioning driver
│
├── paper/
│   └── MTTD_CloudSim_Cricket_WC2023_Research_Paper.pdf # Complete research study paper
│
├── results/
│   ├── cricket/                              # Final results on 2023 Cricket World Cup workload
│   │   ├── cloudsim_results.csv              # Hourly VM allocations and SLA metrics (5 policies)
│   │   ├── cloudsim_run.log                  # Console execution log from CloudSim full run
│   │   ├── predictions_retrain.csv           # Hourly predictions with 2-strike retraining
│   │   ├── predictions_noretrain.csv         # Hourly predictions without retraining
│   │   ├── predictions_daily_retrain.csv     # Predictions from daily retrain + reselect experiment
│   │   ├── predictions_retrain_model_summary.csv # Per-model performance summary table
│   │   └── reselection_results.csv           # Summary of all 6 reselection strategies
│   ├── tables/                               # Synthetic benchmark result tables
│   └── figures/                              # Workload patterns and comparison figures
│
├── arima_baseline.py                         # Phase 2: ARIMA model and Eq. 2-4 metrics
├── svr_rf_models.py                          # Phase 2: SVR and Random Forest regressors
├── xgboost_model.py                          # Phase 2: XGBoost regressor implementation
├── lstm_model.py                             # Phase 2: PyTorch LSTM regressor
├── model_selector.py                         # Phase 3: Model evaluation and selection
├── deployment_manager.py                     # Phase 4: Adaptive threshold state machine
├── predictors.py                             # Streaming inference interface for autoscalers
├── synthetic_workload.py                     # Synthetic workload generator functions
├── sliding_window.py                         # Time-series sliding window transformations
├── run_benchmark.py                          # Synthetic benchmark execution suite
├── mttd_bridge.py                            # MTTD continuous deployment bridge
└── reselection_study.py                      # Reselection-frequency experiment script
```

---

## 7. Installation and Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12 (the final study results were produced on Python 3.12)
- Java Development Kit (JDK 11 or newer; OpenJDK 21 used for published numbers)
- CloudSim 3.0.3 (`cloudsim-3.0.3.jar` from [Cloudslab/cloudsim](https://github.com/Cloudslab/cloudsim/releases/tag/cloudsim-3.0.3))

### Python Environment

```bash
# Clone the repository
git clone https://github.com/garv-codes/ML-workload-prediction.git
cd ML-workload-prediction

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

---

## 8. Usage

### 1. Build the Dataset

Regenerate the canonical hourly workload CSV from raw Wikimedia daily page views:

```bash
python data/build_hourly_workload.py
```

### 2. Run MTTD Continuous Deployment Bridge

Run the full adaptive MTTD pipeline (with 2-strike retraining) and the static baseline:

```bash
# Full MTTD with adaptive retraining (writes to results/cricket/)
python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/cricket/predictions_retrain.csv

# MTTD without retraining (static active model baseline)
python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/cricket/predictions_noretrain.csv --no-retrain
```

### 3. Run CloudSim Datacenter Simulation

Compile and run `ProvisioningSimulation.java` over all 1,080 deployment hours (see [`cloudsim/README.md`](cloudsim/README.md)):

```bash
# Compile
javac -cp cloudsim-3.0.3.jar -d build cloudsim/ProvisioningSimulation.java

# Run from results/cricket/
cd results/cricket
java -cp "../../cloudsim/build;../../cloudsim/cloudsim-3.0.3.jar" mttd.ProvisioningSimulation
```

### 4. Run Reselection-Frequency Study

Evaluate offline and online reselection strategies across different time horizons:

```bash
python reselection_study.py --csv data/cricket_hourly_workload.csv --outdir results/cricket
```

### 5. Run Synthetic Workload Benchmark

```bash
# Benchmark across all 5 candidate models
python run_benchmark.py

# Benchmark original paper models only (ARIMA, SVR, Random Forest)
python run_benchmark.py --original-only
```

---

## 9. Mathematical Formulation

### Error Metrics

**Root Mean Squared Error (RMSE, Eq. 2):**

$$RMSE = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(y_t - \hat{y}_t)^2}$$

**Mean Absolute Error (MAE, Eq. 3):**

$$MAE = \frac{1}{N}\sum_{t=1}^{N}|y_t - \hat{y}_t|$$

**Mean Absolute Percentage Error (MAPE):**

$$MAPE = \frac{1}{N}\sum_{t=1}^{N}\left|\frac{y_t - \hat{y}_t}{y_t}\right| \times 100\%$$

### Performance Score (Eq. 4)

As defined in the paper and implemented in `arima_baseline.performance_score`:

$$\text{Score} = 100 \times \left( 0.5 \times \frac{RMSE}{\sqrt{\frac{1}{N}\sum_{t=1}^{N}y_t^2}} + 0.5 \times \frac{MAE}{\frac{1}{N}\sum_{t=1}^{N}y_t} \right)$$

- Normalized percentage metric: lower values indicate higher accuracy; $0\%$ represents perfect foresight.
- **Active Model Selection:** $\text{Active Model} = \arg\min_i (\text{Score}_i)$.

### Adaptive Retraining Thresholds

| Cycle | Formula | Description |
|---|---|---|
| $t+1$ | $\text{Threshold}_{t+1} = \text{Score}_t \times (1 + \lambda_1) \quad (\lambda_1 = 0.75)$ | Initial tolerance band established after healthy cycle (Eq. 5) |
| $t+2$ | $\text{Threshold}_{t+2} = \text{Score}_{t+1} \times (1 + \lambda_2) \quad (\lambda_2 = 0.50)$ | Tightened threshold applied after first threshold breach (Eq. 6) |

**Retraining Rule:** Retraining is triggered if and only if the Score exceeds the threshold on **two consecutive deployment cycles**.

---

## 10. Experimental Results

### 10.1 Synthetic Workload Benchmarks

Results generated by `run_benchmark.py` across all four synthetic patterns (70/30 train-test split, 100-point test window, window = 5):

| Pattern | Model | RMSE | MAE | MAPE | Score (%) | Best? |
|---|---|---|---|---|---|---|
| uniform | ARIMA | 2.2869 | 1.8859 | 1.42% | 1.51 | [BEST] |
| uniform | SVR | 16.8411 | 13.9486 | 11.49% | 11.11 | |
| uniform | RandomForest | 5.7550 | 5.1867 | 3.96% | 3.95 | |
| uniform | XGBoost | 6.3592 | 5.6850 | 4.31% | 4.35 | |
| uniform | LSTM | 5.4387 | 4.8183 | 3.80% | 3.70 | |
| linear_low | ARIMA | 1.0289 | 0.8201 | 0.44% | 0.49 | [BEST] |
| linear_low | SVR | 97.2096 | 94.1927 | 49.73% | 50.95 | |
| linear_low | RandomForest | 21.1237 | 18.5355 | 9.61% | 10.56 | |
| linear_low | XGBoost | 20.9576 | 18.3463 | 9.51% | 10.46 | |
| linear_low | LSTM | 4.4614 | 4.1254 | 2.17% | 2.29 | |
| linear_high | ARIMA | 1.0375 | 0.8259 | 0.07% | 0.08 | [BEST] |
| linear_high | SVR | 674.8332 | 671.4941 | 54.05% | 54.28 | |
| linear_high | RandomForest | 135.9055 | 118.4355 | 9.29% | 10.25 | |
| linear_high | XGBoost | 136.2235 | 118.8002 | 9.32% | 10.28 | |
| linear_high | LSTM | 27.1023 | 25.8980 | 2.06% | 2.14 | |
| random | ARIMA | 8.2045 | 6.5380 | 3.28% | 3.63 | [BEST] |
| random | SVR | 18.6115 | 12.8222 | 7.00% | 7.73 | |
| random | RandomForest | 8.8420 | 7.2290 | 3.66% | 3.96 | |
| random | XGBoost | 8.5104 | 7.0343 | 3.55% | 3.83 | |
| random | LSTM | 8.2452 | 6.7314 | 3.40% | 3.69 | |

---

### 10.2 Cricket World Cup 2023 Workload Results

#### Workload Summary
- **Total Hits:** 29,957,762 page views across 1,344 hours (56 days)
- **Peak Hour:** 117,612 at 17:00 on 19 Nov 2023 (Final match)
- **Timeline:** Training window (hours 0–239), Testing window (hours 240–263), Deployment window (hours 264–1343 = 1,080 hours / 45 cycles)

#### Initial Testing Phase and Retraining Events
- **Initial Test Phase (Hours 240–263):**
  - RandomForest: **6.49%** $\rightarrow$ **Selected as Active Model**
  - XGBoost: 7.07%
  - SVR: 11.75%
  - LSTM: 17.84%
  - ARIMA: 29.18%
- **Retraining Event at Hour 552 (Cycle 12 breach):**
  - RandomForest: **5.42%** $\rightarrow$ **Retained as Active Model**
  - XGBoost: 7.15%
  - SVR: 7.48%
  - LSTM: 12.22%
  - ARIMA: 33.22%

#### Candidate Model Summary Across All 45 Deployment Cycles

| Model | Mean Score (%) | Mean RMSE | Mean MAE | Cycles | Cycles Best | Overall Score (%) | Hours Active |
|---|---|---|---|---|---|---|---|
| SVR | 72.90 | 6,234.85 | 5,143.76 | 45 | 11 | 22.41 | 0 |
| ARIMA | 34.46 | 11,154.53 | 5,884.14 | 45 | 8 | 30.58 | 0 |
| XGBoost | 54.05 | 7,271.13 | 5,553.95 | 45 | 10 | 30.66 | 0 |
| LSTM | 67.52 | 6,996.31 | 5,376.77 | 45 | 0 | 31.75 | 0 |
| RandomForest | 68.72 | 13,649.96 | 5,945.88 | 45 | 16 | 32.78 | 1,080 |

*(Note: Candidate models in this table are evaluated on the full deployment period; overall Score uses Eq. 4).*

#### MTTD Retraining vs. No-Retraining Comparison

| Mode | RMSE | MAE | Score (%) | Score Reduction |
|---|---|---|---|---|
| **With Retraining** | 13,649.96 | 5,945.88 | **32.78** | **−37.5%** |
| **Without Retraining** | 19,154.94 | 11,397.48 | **52.43** | Baseline |

Retraining achieves a **37.5% reduction** in overall Score (RMSE −28.7%, MAE −47.8%).

---

### 10.3 CloudSim Provisioning Simulation (1,080 Hours)

Full simulation results produced by `ProvisioningSimulation.java` with `MAX_HOURS = 0` (1 datacenter, capacity 150 VMs, 4,000 requests/hour per VM, 10% proactive headroom):

| Policy | VM-hours | Wasted VMh | SLA violations | Under-prov. hours | Avg makespan (s) |
|---|---|---|---|---|---|
| **REACTIVE** | 7,593 | 1,064 | 56 (5.2%) | 56 | 3,217 |
| **MTTD-noRetrain** | 9,018 | 2,623 | 171 (15.8%) | 171 | 2,640 |
| **MTTD-Retrain** | 7,475 | 1,077 | 145 (13.4%) | 145 | 3,036 |
| **ORACLE** | 7,591 | 610 | 0 (0.0%) | 0 | 2,881 |
| **MTTD-Hybrid** | 8,707 | 1,796 | 20 (1.9%) | 20 | 2,586 |

Model usage logged by CloudSim: **RandomForest 1,080 hours (100.0%)**, 0 model switches.

---

### 10.4 Reselection-Frequency Study

Results from `reselection_study.py` evaluating offline selection horizons and daily retraining:

| Strategy | Score (%) | Switches | Models Used | VM-hours | SLA Violations |
|---|---|---|---|---|---|
| **Standard MTTD** | 32.78 | 0 | 1 | 7,475 | 145 |
| **Reselect every 24 h on last 24 h** | 18.55 | 23 | 4 | 7,493 | 138 |
| **Reselect every 24 h on last 72 h** | 16.30 | 11 | 4 | 7,503 | 139 |
| **Reselect every 1 h on last 24 h** | 15.42 | 60 | 5 | 7,481 | 127 |
| **Reselect every 1 h on last 6 h** | 19.93 | 262 | 5 | 7,413 | 120 |
| **Retrain + reselect every 24 h** | 16.13 | 26 | 5 | 7,346 | 124 |

---

### 10.5 Key Observations

- **Active Model Stability vs. Cycle Best:** Random Forest was selected at the initial test phase and again at the single retraining event (hour 552), remaining active for all 1,080 deployment hours. However, the best-scoring candidate varied considerably across individual cycles (Random Forest best in 16 cycles, SVR in 11, XGBoost in 10, ARIMA in 8, LSTM in 0).
- **Impact of Retraining:** Adaptive retraining reduced prediction error from 52.43% to 32.78% (−37.5%), which translated in CloudSim to a 17.1% reduction in total VM-hours (7,475 vs. 9,018) and a 15.2% reduction in SLA-violation hours (145 vs. 171) compared to no retraining.
- **Benefits of Dynamic Reselection:** Re-selecting more frequently (daily with retraining or hourly on trailing 24-hour error) prompted 26 and 60 switches across all five models and lowered the overall Score to 16.13% and 15.42%—substantially outperforming the best single static candidate (SVR at 22.41%).
- **Provisioning Trade-offs:** MTTD-Retrain under-provisioned capacity (total VM-hours were 1.5% below ORACLE, leading to 145 SLA-violation hours vs. 56 for REACTIVE). MTTD-Hybrid effectively protected against traffic spikes, achieving the fewest SLA violations (20, 1.9%) at the expense of 14.7% higher VM-hours than ORACLE.

---

## 11. CloudSim Provisioning Details

The Java CloudSim module (`cloudsim/ProvisioningSimulation.java`) simulates infrastructure provisioning at hourly granularity.

- **Default Execution:** By default, `MAX_HOURS = 0`, simulating the complete 1,080-hour deployment period. Positive integer values can be supplied for quick debugging runs.
- **Command-line Configuration:** Accepts prediction and output paths as arguments (`args[0] = retrain.csv`, `args[1] = noretrain.csv`, `args[2] = output.csv`).
- **Detailed Instructions:** Refer to [`cloudsim/README.md`](cloudsim/README.md) for prerequisite installation, compilation commands, and policy definitions.

---

## 12. Limitations

1. **Synthetic Intra-Day Profile:** Real daily totals are preserved exactly, but the hourly distribution uses a 70/30 match-window split with uniform $\pm 8\%$ noise ($\mathcal{U}(0.92, 1.08)$). Models detecting diurnal periodicity learn this constructed profile.
2. **Noise Floor Bound:** The theoretical lower bound on error under $\pm 8\%$ uniform noise is 4.06%.
3. **Traffic Proxy:** Wikipedia page views capture human interest rather than backend cloud server CPU/memory transactions.
4. **Non-Comparability to Paper Numbers:** Results are evaluated on the 2023 Cricket World Cup workload rather than the 1998/2018 FIFA logs.
5. **Single Tournament & Seed:** Benchmarks reflect one tournament and a single noise seed (seed 42).
6. **Cross-Validation Scheme:** Hyperparameter tuning via `GridSearchCV` uses standard K-fold cross-validation rather than time-ordered rolling splits.
7. **Non-Seasonal ARIMA:** The ARIMA implementation is non-seasonal, with order limits `max_p = 5, max_q = 5`.

---

## 13. Research Paper

The complete research paper describing this study, methodology, and findings is available in the repository:

- **PDF:** [`paper/MTTD_CloudSim_Cricket_WC2023_Research_Paper.pdf`](paper/MTTD_CloudSim_Cricket_WC2023_Research_Paper.pdf)

### Citation

```bibtex
@article{mishra2026mttd,
  title   = {Reproducing and Extending the MTTD Dynamic Model-Selection Framework: Five ML Models on a 2023 Cricket World Cup Workload with CloudSim Evaluation},
  author  = {Mishra, Abhinav and Gujral, Garv and Yadav, Shashank},
  journal = {Netaji Subhas University of Technology Research Report},
  year    = {2026},
  address = {New Delhi 110075, India}
}
```

---

## 14. References

1. Nashaat, M., et al. "Dynamic machine learning approach for workload prediction in cloud environments." *Scientific Reports*, vol. 16, no. 1, 10983, 2026. DOI: [10.1038/s41598-026-40777-z](https://www.nature.com/articles/s41598-026-40777-z)
2. Calheiros, R. N., Ranjan, R., Beloglazov, A., De Rose, C. A., & Buyya, R. "CloudSim: a toolkit for modeling and simulation of cloud computing environments and evaluation of resource provisioning algorithms." *Software: Practice and Experience*, vol. 41, no. 1, pp. 23–50, 2011. DOI: [10.1002/spe.995](https://doi.org/10.1002/spe.995)
3. Wikimedia Foundation. *Pageviews Analysis Tool*. <https://pageviews.wmcloud.org/>
4. Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *ACM KDD*, 2016.
5. Hochreiter, S., & Schmidhuber, J. "Long Short-Term Memory." *Neural Computation*, 1997.

---

## 15. License

This repository is licensed under the [MIT License](LICENSE).
