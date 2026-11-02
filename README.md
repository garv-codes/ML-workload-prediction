# Machine Learning Workload Prediction in Cloud Environments

**Dynamic Multi-Model Framework with Adaptive Retraining (MTTD)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports%202026-blue.svg)](https://www.nature.com/articles/s41598-026-40777-z)

---

## 1. Overview

Cloud computing platforms rely on proactive autoscaling to maintain Service Level Agreements (SLAs) while minimising expensive resource over-provisioning. Traditional reactive autoscalers respond only after demand spikes occur, incurring latency penalties and SLA violations.

This repository re-implements the **MTTD (Monitor–Train–Test–Deploy)** framework introduced by [Nashaat et al., *Sci. Rep.* 16, 10983 (2026)](https://www.nature.com/articles/s41598-026-40777-z) and extends it with additional models, evaluation metrics, a semi-synthetic multi-day workload trace, and a CloudSim provisioning simulation.

### Objectives

| # | Objective | Description |
|---|---|---|
| 1 | **Paper Reproduction** | Implement all four MTTD phases using ARIMA, SVR, and Random Forest following Eq. 2-6 as published. |
| 2 | **Extended Framework** | Introduce **XGBoost** (gradient boosting vs. bagging) into the candidate pool, evaluate **MAPE** as a scale-invariant metric, expand candidate pool capacity to 5 models, and simulate infrastructure provisioning in CloudSim. |
| 3 | **Deep Learning Comparison** | Evaluate LSTM recurrent architectures (with automatic neural network MLP fallback when deep learning frameworks are absent) alongside statistical and ensemble methods. |
| 4 | **Semi-Synthetic Validation** | Evaluate the framework on an hourly workload trace derived from real-world Wikimedia daily page views (see [§4 Dataset](#4-dataset)). |
| 5 | **Deployment API** | Provide a modular streaming predictor interface (`predictors.py`) suitable for integration into external autoscaling controllers. |

---

## 2. Framework Architecture

The MTTD framework operates continuously through four coordinated phases (corresponding to Fig. 2 in Nashaat et al., 2026):

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 1 — Monitoring                                                    │
│    • Log incoming request traffic and aggregate into time-series data    │
│    • Extract sliding window features (WINDOW = 5 for synthetic, 24 real) │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 2 — Training                                                      │
│    • Fit candidate models on recent training history                     │
│    • Model pool: ARIMA, SVR (RBF), Random Forest, XGBoost, LSTM / MLP    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 3 — Testing and Evaluating                                        │
│    • Evaluate forecasts across the test window                           │
│    • Compute Eq. 4 Score = 100 * (w1 * RMSE / RMS(y) + w2 * MAE / Mean(y))│
│    • Select Active Model = argmin(Score)                                 │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 4 — Continuous Deployment                                         │
│    • Active model issues one-step-ahead forecasts each timestep          │
│    • Evaluate Score at cycle boundary against adaptive threshold         │
│    • Strike 1: Score > Threshold → tighten threshold (Eq. 6)             │
│    • Strike 2: Two consecutive breaches → trigger Phase 2/3 retraining   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Provenance: What is from the Paper vs. Our Extensions

| Aspect | Paper (Nashaat et al., 2026) | This Repository |
|---|---|---|
| **Core Framework** | 4-phase MTTD pipeline (Monitor, Train, Test, Deploy) | Faithful reproduction of the 4-phase loop |
| **Model Selection & Thresholds** | Equations 2–6 as published | Eq. 2–6 implemented; threshold tightening interpretation documented |
| **Models Evaluated in Paper** | ARIMA, SVR, Random Forest, and **LSTM** (Section "Evaluating the model with changes in the used algorithms") | ARIMA, SVR, Random Forest, **LSTM** (multi-backend PyTorch / TensorFlow / MLP fallback) |
| **Novel Model Extension** | Not evaluated | **XGBoost Regressor** (gradient-boosted trees) |
| **Candidate Pool Size** | Capped at 3 active candidates | 5 candidate models evaluated simultaneously |
| **Evaluation Metrics** | RMSE, MAE, and Eq. 4 combined Score | RMSE, MAE, Eq. 4 Score, plus scale-invariant **MAPE** |
| **Datasets** | FIFA World Cup 1998 and 2018 HTTP server access logs | 4 synthetic benchmark patterns + semi-synthetic Cricket Hourly Workload |
| **Simulation Study** | Conceptual formulation | Java CloudSim provisioning simulation module (`cloudsim/`) |

> **Note on Datasets:** The paper's FIFA access logs are available from the original authors at
> <https://github.com/walidmoussa2010/Workload-prediction->.

---

## 4. How the MTTD Loop Runs (`mttd_bridge.py`)

The end-to-end continuous deployment simulation is implemented in `mttd_bridge.py`:

- **Constants:** `WINDOW = 24` hours, `TRAIN_LEN = 240` hours (10 days), `TEST_LEN = 24` hours (1 day), `CYCLE = 24` hours (1 day), `lambda1 = 0.75`, `lambda2 = 0.50`.
- **Forecast Step:** The designated active model forecasts one hour ahead at every timestep.
- **Cycle Boundary:** At the conclusion of every 24-hour deployment cycle, the cycle Score is calculated using Eq. 4.
- **Threshold Check:** The score is compared against the active threshold:
  - If Score $\le$ Threshold: Cycle is healthy; strike count resets to 0; threshold refreshes to $\text{Score} \times (1 + \lambda_1)$ (Eq. 5).
  - If Score $>$ Threshold (Strike 1): First breach recorded; threshold tightens to $\text{Score} \times (1 + \lambda_2)$ (Eq. 6); active model remains deployed.
  - If Score $>$ Threshold on the next cycle (Strike 2): **Consecutive breach triggers retraining.**
- **Retraining Procedure:** When triggered at a cycle boundary, all 5 candidate models are retrained on the most recent 264 hours (`TRAIN_LEN + TEST_LEN`), re-tested on the trailing 24 hours (`TEST_LEN`), and the model with the minimum test Score becomes the new active model.
- **Stability Guarantee:** A single anomalous cycle never triggers retraining. Model switching can occur only at 24-hour cycle boundaries following two consecutive breaches.
- **Interpretation Note:** In the published paper, Equations 5 and 6 are underspecified (the values reported in Table 12 do not strictly satisfy them as literally written). In this implementation, the threshold refreshes from the latest healthy score (Eq. 5) and tightens upon the first breach (Eq. 6).
- **Implementation Note:** In `mttd_bridge.py`, all 5 candidate models issue forecasts each cycle for logging, comparison, and analysis. The paper's low-overhead justification assumes only the single active model executes; therefore no computational overhead reduction is claimed for this logging run.

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

### 5.2 Cricket Hourly Workload (Semi-Synthetic)

A multi-day trace used to validate the framework over 56 days of workload. Provenance and construction details are documented in [`data/README.md`](data/README.md).

| Fact | Value |
|---|---|
| **Source** | Daily page views exported from Wikimedia Pageviews Analysis tool (`pageviews.wmcloud.org`). Article name, project, access and agent filters: `TODO(owner)` |
| **Period** | 2023-10-01 00:00:00 to 2023-11-25 23:00:00 (**56 days**) |
| **Rows** | 1,344 hourly observations (56 days × 24 hours), schema `Timestamp,Hits` |
| **Real Component** | 56 daily page-view totals. Preserved **exactly** in the hourly file (daily sum of hourly hits = daily total) |
| **Synthetic Component** | Intra-day distribution: 70% of daily total spread evenly over 14:00–21:59 (8 hours, 8.75% each); 30% spread evenly over remaining 16 hours (1.875% each). Flat distribution within bands. Multiplied by uniform noise in `[0.92, 1.08]` (uniform ±8%), renormalised to match daily total, rounded to integers (residual to peak hour). Seed 42 |
| **Statistics** | Total hits: 29,957,762; Mean: 22,290.0; Min: 630; Max: 117,612; Day 1: 111,272; Day 56: 36,908 |
| **File SHA-256** | `0411f7b72cab109e31d0eed31da0c385c831ff85796dea49905b8b0926140c45` |
| **Oracle Noise Floor** | **4.06%** (theoretical minimum Score under ±8% uniform noise) |

> **Context:** The paper's FIFA access logs reflect minute-level HTTP requests. This project utilizes a modern public data source with real daily totals. The hourly intra-day structure is modelled rather than measured, and page views serve as an indirect proxy for server load.

---

## 6. Project Structure

```
ML-workload-prediction/
├── README.md                          # Root framework documentation
├── requirements.txt                   # Pinned project dependencies
├── LICENSE                            # MIT License
├── .gitattributes                     # Binary preservation for CSV datasets
├── .gitignore                         # Ignore patterns for bytecode, venv, caches
│
├── data/
│   ├── README.md                      # Dataset provenance, facts, and verification
│   ├── cricket_hourly_workload.csv    # Canonical 1,344-row hourly dataset
│   ├── build_hourly_workload.py       # Canonical dataset generator script
│   ├── verify_dataset.py              # Verification suite (hash, sums, noise floor)
│   └── raw/
│       └── daily_totals_recovered.csv # 56 exact daily totals recovered from canonical CSV
│
├── cloudsim/
│   ├── README.md                      # CloudSim simulation guide, policies, metrics
│   └── ProvisioningSimulation.java    # Java CloudSim datacenter provisioning driver
│
├── tools/
│   └── results_to_markdown.py         # Automated markdown table generator from results
│
├── tests/
│   ├── test_dataset.py                # Dataset integrity verification tests
│   ├── test_deployment_manager.py     # State machine tests for adaptive thresholding
│   └── test_metrics.py                # Mathematical formulation tests (Eq. 2-4)
│
├── results/
│   ├── README.md                      # Index of all result artifacts and schemas
│   ├── run_manifest.json              # Run metadata, versions, constants, and fallbacks
│   ├── predictions_retrain.csv        # Hourly forecasts with 2-strike retraining
│   ├── predictions_noretrain.csv      # Hourly forecasts without retraining
│   ├── predictions_*_model_scores.csv # Per-cycle candidate scores and thresholds
│   ├── predictions_*_model_summary.csv# 45-cycle aggregate candidate performance
│   ├── figures/                       # Plots, prediction graphs, comparison charts
│   ├── tables/                        # Synthetic benchmark result CSV tables
│   └── logs/                          # Captured console execution logs
│
├── synthetic_workload.py              # Phase 1: Synthetic workload generators
├── sliding_window.py                  # Phase 1: Sliding window time-series transform
├── arima_baseline.py                  # Phase 2: ARIMA model and Eq. 2-4 metrics
├── svr_rf_models.py                   # Phase 2: SVR and Random Forest with grid search
├── xgboost_model.py                   # Phase 2: XGBoost Regressor implementation
├── lstm_model.py                      # Phase 2: Multi-backend LSTM with MLP fallback
├── model_selector.py                  # Phase 3: Model evaluation and selection
├── deployment_manager.py              # Phase 4: Adaptive threshold state machine
├── predictors.py                      # Streaming inference interface for autoscalers
├── run_benchmark.py                   # Benchmark suite for synthetic patterns
└── mttd_bridge.py                     # Continuous deployment simulation bridge
```

---

## 7. Installation and Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Standard virtual environment recommended

### Setup

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

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 8. Usage

### 1. Verify Dataset Integrity

```bash
python data/verify_dataset.py
```

### 2. Execute Automated Unit Tests

```bash
pytest
```

### 3. Run Synthetic Workload Benchmark

```bash
# Full benchmark across all 5 candidate models
python run_benchmark.py

# Paper reproduction benchmark (ARIMA, SVR, Random Forest only)
python run_benchmark.py --original-only
```

### 4. Run MTTD Continuous Deployment Bridge

```bash
# Full MTTD with adaptive 2-strike retraining
python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_retrain.csv

# Baseline MTTD without retraining (static initial active model)
python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out results/predictions_noretrain.csv --no-retrain
```

### 5. Generate Markdown Result Tables

```bash
python tools/results_to_markdown.py
```

### 6. CloudSim Datacenter Simulation

See [`cloudsim/README.md`](cloudsim/README.md) for compilation and execution instructions using CloudSim 3.0.3.

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

$$\text{Score} = 100 \times \left( w_1 \frac{RMSE}{\sqrt{\frac{1}{N}\sum_{t=1}^{N}y_t^2}} + w_2 \frac{MAE}{\frac{1}{N}\sum_{t=1}^{N}y_t} \right)$$

- Default parameters: $w_1 = 0.5$, $w_2 = 0.5$.
- Normalized percentage metric: lower values indicate higher accuracy; $0\%$ indicates perfect foresight.
- **Active Model Selection:** $\text{Active Model} = \arg\min_i (\text{Score}_i)$.

### Adaptive Retraining Thresholds

| Cycle | Formula | Description |
|---|---|---|
| $t+1$ | $\text{Threshold}_{t+1} = \text{Score}_t + \lambda_1 \times \text{Score}_t \quad (\lambda_1 = 0.75)$ | Initial tolerance band established after healthy cycle (Eq. 5) |
| $t+2$ | $\text{Threshold}_{t+2} = \text{Score}_{t+1} + \lambda_2 \times \text{Score}_{t+1} \quad (\lambda_2 = 0.50)$ | Tightened threshold applied after first threshold breach (Eq. 6) |

**Retraining Rule:** Retraining is triggered if and only if the Score exceeds the threshold on **two consecutive deployment cycles**.

---

## 10. Experimental Results

### 10.1 Synthetic Workload Benchmarks

Results generated by `run_benchmark.py` across all four synthetic patterns (Duration: 697 minutes, 70/30 train-test split, 100-point test window, window = 5):

**Grand Summary Across Workload Patterns**

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

### 10.2 MTTD Bridge Results (Cricket Hourly Workload)

Results generated by `mttd_bridge.py` on the canonical 1,344-hour trace (240-hour initial train, 24-hour test, 45 deployment cycles = 1,080 hours):

**Per-Model Summary (With Retraining)**

| Model | Mean Score (%) | Mean RMSE | Mean MAE | Cycles | Cycles Best | Overall Score (%) | Hours Active |
|---|---|---|---|---|---|---|---|
| SVR | 72.90 | 6,234.85 | 5,143.76 | 45 | 9 | 22.41 | 0 |
| NeuralNet-MLP | 67.52 | 6,996.31 | 5,376.77 | 45 | 2 | 24.65 | 0 |
| ARIMA | 34.46 | 11,154.53 | 5,884.14 | 45 | 8 | 30.57 | 0 |
| XGBoost | 54.05 | 7,271.13 | 5,553.95 | 45 | 10 | 30.66 | 0 |
| RandomForest | 53.37 | 7,714.48 | 5,945.88 | 45 | 16 | 32.78 | 1,080 |

*(Note: Fixed-model rows in this table are also retrained at MTTD's retraining events; they do not represent "train once" static baselines.)*

**Retrain Events**

| Phase | Start Hour | Model | Score (%) | Threshold (%) |
|---|---|---|---|---|
| Deploy12 | 528 | SVR | 19.61 | 31.127 |
| Deploy12 | 528 | RandomForest | 97.481 | 31.127 |
| Deploy12 | 528 | ARIMA | 29.406 | 31.127 |
| Deploy12 | 528 | XGBoost | 63.166 | 31.127 |
| Deploy12 | 528 | NeuralNet-MLP | 13.794 | 31.127 |

**MTTD Retraining vs. No-Retraining Comparison**

| Mode | RMSE | MAE | Score (%) |
|---|---|---|---|
| With Retraining | 13,649.96 | 5,945.88 | 32.78 |
| Without Retraining | 19,154.94 | 11,397.48 | 52.43 |

**MTTD Dynamic Active Model vs. Best Fixed Model**

| Strategy | Overall Score (%) | Notes |
|---|---|---|
| MTTD (dynamic) | 32.78 | RandomForest active across 1,080 hours |
| Best Fixed (SVR) | 22.41 | Lowest overall Score (9 cycles best) |

- **Comparison Outcome:** MTTD with retraining (32.78%) substantially improves over MTTD without retraining (52.43%), demonstrating the necessity of adaptive retraining. However, MTTD does not outperform the best individual model (SVR at 22.41% or NeuralNet-MLP at 24.65%) over the full deployment period. This occurs because the initial model selection in Phase 3 picked Random Forest based on a single 24-hour test window, and the two-strike thresholding maintained Random Forest through cycles where SVR or ARIMA yielded lower error.
- **Oracle Noise Floor:** **4.06%**. Theoretical limit imposed by the ±8% uniform intra-day noise.
- **CloudSim Provisioning Evaluation:** `TODO(owner)`: execution pending provision of external CloudSim 3.0.3 and Apache Commons Math 3 JAR dependencies.

---

## 11. Limitations

1. **Synthetic Hourly Structure:** Intra-day hourly distribution (70/30 peak/off-peak split, uniform ±8% noise) is synthetic. Models detecting the 24-hour periodicity are learning the structure constructed by the generator.
2. **Noise Floor Bound:** The 4.06% oracle noise floor is the lower limit of achievable accuracy on the semi-synthetic dataset.
3. **Proxy Metric:** Wikimedia page views represent user web traffic interest rather than back-end cloud server HTTP transactions.
4. **Non-comparability to Paper Tables:** Results are evaluated on the semi-synthetic trace and synthetic benchmark generators, and are not directly comparable to the paper's FIFA access log tables.
5. **ARIMA Parameter Limits:** Grid search is constrained to `max_p = max_q = 5`, whereas the original paper explored orders up to `(9, 1, 10)`.
6. **Cross-Validation Scheme:** Model hyperparameter tuning uses standard K-fold cross-validation rather than time-ordered rolling splits.
7. **Single Realization:** Benchmarks represent single deterministic runs (seed 42) rather than multi-trial statistical distributions.
8. **Selection Window Scope:** Model selection occurs across a single 24-hour evaluation cycle.
9. **Historical Trace Versions:** Earlier unverified data releases (such as commit `808162d` with trailing text banners) produced superseded results that have been replaced.

---

## 12. Deployment Integration

The `predictors.py` module exposes a streaming interface for online deployment:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from deployment_manager import DeploymentManager
from predictors import make_rf_predictor

# Illustrative usage example with mock training history
history = np.random.uniform(1000, 5000, size=50).tolist()
X = np.array([history[i:i+5] for i in range(len(history)-5)])
y = np.array(history[5:])
model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)

# Initialize predictor and deployment manager
manager = DeploymentManager(active_model_name="RandomForest", initial_score=5.0)
predictor = make_rf_predictor(model, window=5, initial_history=history[-5:])

# Issue one-step-ahead prediction
next_val = predictor.predict_next()
actual_observed = 2500.0
predictor.observe(actual_observed)

# Cycle check at 24-hour boundary
status = manager.run_cycle(actual_values=[actual_observed], predicted_values=[next_val])
if status["retrain_triggered"]:
    pass  # Retrain candidate models
```

---

## 13. References

1. Nashaat, M., et al. "Dynamic machine learning approach for workload prediction in cloud environments." *Scientific Reports*, vol. 16, no. 1, 10983, 2026. DOI: [10.1038/s41598-026-40777-z](https://www.nature.com/articles/s41598-026-40777-z)
2. Calheiros, R. N., Ranjan, R., Beloglazov, A., De Rose, C. A., & Buyya, R. "CloudSim: a toolkit for modeling and simulation of cloud computing environments and evaluation of resource provisioning algorithms." *Software: Practice and Experience*, vol. 41, no. 1, pp. 23–50, 2011. DOI: [10.1002/spe.995](https://doi.org/10.1002/spe.995)
3. Wikimedia Foundation. *Pageviews Analysis Tool*. <https://pageviews.wmcloud.org/>
4. Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *ACM KDD*, 2016.
5. Hochreiter, S., & Schmidhuber, J. "Long Short-Term Memory." *Neural Computation*, 1997.

---

## 14. License

This repository is licensed under the [MIT License](LICENSE).
