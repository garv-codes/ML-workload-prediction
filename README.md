# ☁️ Machine Learning Workload Prediction in Cloud Environments
### *Dynamic Multi-Model Framework with Adaptive Retraining (MTTD)*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Research Reproduction](https://img.shields.io/badge/reproduction-Scientific%20Reports%202026-success.svg)](https://www.nature.com/articles/s41598-026-40777-z)
[![Novelty](https://img.shields.io/badge/novelty-XGBoost%20%7C%20LSTM%20%7C%20MAPE-orange.svg)](#-novelty-contributions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Course:** Cloud Computing / Machine Learning Project (3rd Year B.Tech / B.E.)  
> **Original Paper:** *"A dynamic machine learning approach for workload prediction in cloud environments"*  
> **Authors:** Nashaat et al., *Scientific Reports* (Nature Portfolio), 2026  
> **Paper DOI:** [`10.1038/s41598-026-40777-z`](https://www.nature.com/articles/s41598-026-40777-z)

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [MTTD Framework Architecture](#-mttd-framework-architecture)
3. [Novelty Contributions](#-novelty-contributions)
4. [Project Structure](#-project-structure)
5. [Installation & Setup](#-installation--setup)
6. [Running the Experiments](#-running-the-experiments)
7. [Mathematical Formulation](#-mathematical-formulation)
8. [Experimental Results & Reproduction](#-experimental-results--reproduction)
9. [Integration Guide for Teammates (Cloud / Autoscaling)](#-integration-guide-for-teammates)
10. [Viva & Examination Cheatsheet](#-viva--examination-cheatsheet)

---

## 📖 Project Overview

Modern cloud platforms (Kubernetes, AWS Auto Scaling, Azure VM Scale Sets) require proactive autoscaling to maintain Service Level Agreements (SLAs) while minimizing resource over-provisioning costs. Reactive threshold-based autoscaling responds *after* traffic spikes, causing latency spikes and SLA penalties.

This repository implements the **MTTD (Model Training, Testing, and Deployment)** dynamic workload prediction framework from Nashaat et al. (2026), enhanced with **novel ML/DL architectures (XGBoost, LSTM)** and a modular streaming inference API for cloud orchestration systems.

### Key Objectives
1. **Faithful Paper Reproduction:** Implement all 4 phases of MTTD with exact mathematical formulations for ARIMA, SVR, Random Forest, Performance Scoring ($S_i$), and the two-tier adaptive retraining threshold ($\lambda_1, \lambda_2$).
2. **Novelty Extension:** Bridge gaps in the original paper by introducing:
   - **XGBoost (Extreme Gradient Boosting)** to contrast boosting against Random Forest bagging.
   - **LSTM (Long Short-Term Memory)** to evaluate deep recurrent modeling on sequential temporal dependencies.
   - **MAPE (Mean Absolute Percentage Error)** for relative scale-invariant accuracy assessment.
3. **Cloud Autoscaler Decoupling:** Provide a clean `predictors.py` interface enabling teammates to plug in trained models directly into autoscaling controllers or load balancers.

---

## 🏗️ MTTD Framework Architecture

The framework operates across 4 continuous phases:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Workload Identification & Preprocessing                       │
│  - Synthetic & Real Traces (Uniform, Linear Low, Linear High, Random)  │
│  - Sliding Window Feature Extraction (window = 5 timesteps)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Candidate Model Training                                      │
│  - Paper Baseline: ARIMA(p,d,q), SVR (RBF Kernel), Random Forest       │
│  - Novelty Models: XGBoost Regressor, PyTorch/TF LSTM                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Testing & Dynamic Model Selection                             │
│  - Compute Root Mean Squared Error (RMSE) & Mean Absolute Error (MAE)  │
│  - Compute Nashaat Performance Score: Score = sqrt(RMSE^2 + MAE^2)     │
│  - Select Active Model = argmin(Score)                                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Continuous Deployment & Adaptive Retraining                   │
│  - Deploy Active Model for streaming one-step-ahead forecasting        │
│  - Strike 1: Score_t > Threshold_(t+1) -> Tighten threshold (lambda_2) │
│  - Strike 2: Score_(t+1) > Threshold_(t+2) -> Trigger RETRAINING       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Novelty Contributions

The original 2026 paper evaluates only ARIMA, SVR, and Random Forest. Our extensions provide tangible academic value:

| Feature | Original Paper (Nashaat et al.) | Our Extended Implementation | Academic Justification |
| :--- | :--- | :--- | :--- |
| **Ensemble Variety** | Bagging only (Random Forest) | **Random Forest + XGBoost** | Evaluates gradient boosting vs bagging under abrupt workload variance |
| **Deep Learning** | None (explicitly avoided) | **PyTorch / TensorFlow LSTM** | Tests whether recurrent memory cells capture non-linear temporal dynamics better on volatile traces |
| **Error Metrics** | RMSE, MAE, Score | **RMSE, MAE, MAPE, Score** | MAPE provides percentage deviation relative to current scale |
| **Runtime Portability** | Custom script | **Defensive CPU-optimized fallback** | Runs seamlessly across standard CPU environments and Python 3.12 |
| **API Decoupling** | Monolithic evaluation | **`predictors.py` Streaming API** | Clean separation of ML pipeline and teammate cloud deployment code |

---

## 📂 Project Structure

```bash
cloud-computing/
├── .gitignore                   # Ignores caches, checkpoints, and generated figures
├── requirements.txt             # Project dependencies (pinned with compatible versions)
├── README.md                    # Comprehensive documentation and integration guide
│
├── synthetic_workload.py        # Phase 1: Generates the 4 workload patterns (Fig. 1)
├── sliding_window.py            # Phase 1: Sliding window time-series transform
│
├── arima_baseline.py            # Phase 2: ARIMA auto-fit & evaluation (Eq. 1-4)
├── svr_rf_models.py             # Phase 2: SVR (RBF) & Random Forest with GridSearchCV
├── xgboost_model.py             # Novelty: XGBoost regressor with hyperparameter tuning
├── lstm_model.py                # Novelty: Lightweight LSTM with PyTorch / TF backends
│
├── model_selector.py            # Phase 3: Evaluates candidate models & selects active model
├── deployment_manager.py        # Phase 4: Two-tier adaptive threshold & retraining triggers
├── predictors.py                # Online streaming inference wrappers for cloud orchestration
│
├── run_benchmark.py             # Complete benchmark script reproducing Tables 5-8 & plots
└── results/                     # Auto-generated outputs (CSVs, figures, summary charts)
    ├── figures/
    └── tables/
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Recommended: Virtual environment (`venv`)

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

# Install dependencies
pip install -r requirements.txt
```

> **Note on Deep Learning:** If using PyTorch on Windows without a GPU, install the lightweight CPU wheel:  
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## 🚀 Running the Experiments

### 1. Full Benchmark (Reproduction + Novelty)
Runs all 5 candidate models across all 4 workload patterns, computes metrics, selects active models, and exports comparison charts and CSV tables:
```bash
python run_benchmark.py
```

### 2. Paper-Only Reproduction
To run strictly the original 3 models (ARIMA, SVR, Random Forest) matching Tables 5–8 of the paper:
```bash
python run_benchmark.py --original-only
```

### 3. Individual Component Execution
Each file is self-contained with a standalone demonstration:

```bash
# Test synthetic workload generators
python synthetic_workload.py

# Test ARIMA baseline
python arima_baseline.py

# Test SVR and Random Forest
python svr_rf_models.py

# Test XGBoost novelty model
python xgboost_model.py

# Test LSTM novelty model
python lstm_model.py

# Test Phase 3 model selection
python model_selector.py

# Test Phase 4 adaptive threshold & regime shift retraining
python deployment_manager.py
```

---

## 📐 Mathematical Formulation

### 1. Error Metrics (Equations 1–3)

- **Root Mean Squared Error (RMSE):**
  $$RMSE = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(y_t - \hat{y}_t)^2}$$

- **Mean Absolute Error (MAE):**
  $$MAE = \frac{1}{N}\sum_{t=1}^{N}|y_t - \hat{y}_t|$$

- **Mean Absolute Percentage Error (MAPE - Novelty):**
  $$MAPE = \frac{1}{N}\sum_{t=1}^{N}\left|\frac{y_t - \hat{y}_t}{y_t}\right| \times 100\%$$

### 2. Performance Score (Equation 4)
The paper defines a single combined metric to rank models:
$$\text{Score}_i = \sqrt{RMSE_i^2 + MAE_i^2}$$
$$\text{Active Model} = \arg\min_i (\text{Score}_i)$$

### 3. Continuous Deployment Thresholds (Equations 5–6)
When a model is deployed, its threshold dynamically adapts based on its performance:

- **Cycle $t+1$ (First Threshold):**
  $$\text{Threshold}_{t+1} = \text{Score}_t + \lambda_1 \times \text{Score}_t \quad (\lambda_1 = 0.75)$$

- **Cycle $t+2$ (Tightened Threshold after 1 Strike):**
  $$\text{Threshold}_{t+2} = \text{Score}_{t+1} + \lambda_2 \times \text{Score}_{t+1} \quad (\lambda_2 = 0.50)$$

- **Retraining Condition:** If $\text{Score} > \text{Threshold}$ on two consecutive cycles, retraining and model reselection are immediately triggered.

---

## 📊 Experimental Results & Reproduction

### Summary Across Workload Patterns

| Workload Pattern | Characteristics | Paper Best Model | Our Extended Winner | Why? |
| :--- | :--- | :--- | :--- | :--- |
| **Uniform / Bell** | Smooth rise to peak, smooth descent | **SVR** | **SVR / LSTM** | RBF kernel and LSTM smoothly interpolate non-linear parabolic curves without step artifacts. |
| **Linear Low Rate** | Steady slope (0.35 req/s/min) | **Random Forest** | **Random Forest / XGBoost** | Tree-based models partition monotonous intervals with minimal bias. |
| **Linear High Rate** | Steep slope (2.31 req/s/min) | **Random Forest** | **XGBoost** | Gradient boosting refines residual errors on sharp gradients better than bagging. |
| **Random Workload** | High volatility, low autocorrelation | **Retraining Loop** | **LSTM / ARIMA** | LSTM filters noise via forget gates; frequent retraining stabilizes drift. |

All generated plots and tables are saved directly into the `results/` directory:
- `results/figures/fig1_workload_patterns.png`
- `results/figures/predictions_<pattern>.png`
- `results/figures/summary_comparison.png`
- `results/tables/table_<pattern>.csv`
- `results/tables/grand_summary.csv`

---

## 🤝 Integration Guide for Teammates

If your teammates are building the Cloud Autoscaling Controller (e.g., using Kubernetes HPA, Docker, Flask, or AWS SDK), this section explains how to consume this ML module cleanly.

### High-Level Interface Flow
Teammates do **not** need to worry about sliding window transforms or PyTorch/sklearn internals. They use `DeploymentManager` and `predictors.py`:

```python
# =====================================================================
# Example Teammate Autoscaler Script: autoscaler_controller.py
# =====================================================================
from deployment_manager import DeploymentManager
from predictors import make_rf_predictor, make_xgboost_predictor, make_svr_predictor

# 1. Initialize DeploymentManager with initial Phase 3 score
manager = DeploymentManager(active_model_name="RandomForest", initial_score=1.42)

# 2. In your cloud event loop (e.g., running every 1 minute):
def on_minute_tick(actual_request_count, fitted_model, window_history):
    # a. Predict next minute's traffic using unified predictor
    predictor = make_rf_predictor(fitted_model, window=5, initial_history=window_history)
    predicted_next_min = predictor.predict_next()
    
    # b. Compute scaling decision for Kubernetes / Cloud VM
    # Formula: Target Replicas = ceil(predicted_traffic / capacity_per_pod)
    pods_needed = int(np.ceil(predicted_next_min / 50.0))
    # scale_kubernetes_deployment(replicas=pods_needed)
    
    # c. At the end of each deployment cycle (e.g., every 10 mins), feed cycle data:
    cycle_status = manager.run_cycle(actual_values=[...], predicted_values=[...])
    
    if cycle_status["retrain_triggered"]:
        print("⚠️ Warning: Workload drift detected! Triggering Model Retraining...")
        # Trigger Phase 2 & 3 reselection
```

---

## 🎓 Viva & Examination Cheatsheet

### Frequently Asked Questions by Professors

1. **Q: Why combine RMSE and MAE into a single Performance Score?**  
   *A:* RMSE heavily penalizes large outlier errors (due to squaring), whereas MAE provides a robust linear average of typical error magnitude. Combining both via Euclidean norm $\sqrt{RMSE^2 + MAE^2}$ balances sensitivity to extreme traffic spikes and steady-state precision.

2. **Q: Why does SVR perform best on Uniform (Bell-shaped) workloads?**  
   *A:* The Radial Basis Function (RBF) kernel maps temporal features into infinite-dimensional Hilbert space, naturally fitting smooth parabolic curvatures without the step-like piecewise approximations typical of decision trees.

3. **Q: Why does the framework use a two-strike threshold instead of retraining on the first error spike?**  
   *A:* Cloud environments frequently experience transient network noise or momentary bursts. Immediate retraining would cause thrashing, high compute overhead, and model degradation. A two-strike mechanism with a tightened second threshold ($\lambda_2 = 0.50$) filters temporary outliers while swiftly catching genuine regime changes.

4. **Q: What is the novelty of adding XGBoost and LSTM?**  
   *A:* The 2026 paper omitted gradient boosting and deep recurrent models. XGBoost tests sequential gradient-based tree boosting (reducing bias on steep linear climbs), while LSTM tests hidden recurrence for sequence modeling without relying purely on fixed lag features.

---

## 📜 References
- Nashaat, M., et al. "A dynamic machine learning approach for workload prediction in cloud environments." *Scientific Reports*, vol. 16, no. 1, 2026. DOI: [10.1038/s41598-026-40777-z](https://www.nature.com/articles/s41598-026-40777-z).
- Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *ACM KDD*, 2016.
- Hochreiter, S., & Schmidhuber, J. "Long Short-Term Memory." *Neural Computation*, 1997.
