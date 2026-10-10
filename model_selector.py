"""
Model selection logic — Phase 3 ("Testing and Model Selection") of the MTTD
framework, following Fig. 3's pseudocode (steps 5-13):

  For each candidate model:
      train it on the training window
      test it on the test window
      compute its Performance Score
  Select the model with the LOWEST score as the active model.

Extended with LSTM and XGBoost as novelty additions to the original paper's
ARIMA, SVR, and Random Forest candidates.
"""

import pandas as pd

from arima_baseline import evaluate_arima_on_workload
from svr_rf_models import evaluate_svr_on_workload, evaluate_rf_on_workload
try:
    from xgboost_model import evaluate_xgboost_on_workload
except ImportError:
    evaluate_xgboost_on_workload = None

try:
    from lstm_model import evaluate_lstm_on_workload
except ImportError:
    evaluate_lstm_on_workload = None


def evaluate_all_models(df, train_frac=0.7, test_len=100, window=5,
                         value_col="avg_requests_per_min",
                         include_novelty=True):
    """Runs all candidate models on the same workload/test split
    and returns a dict of results keyed by model name.

    Args:
        include_novelty: if True (default), also runs LSTM and XGBoost.
                         Set False to reproduce the original paper's 3-model
                         comparison only.
    """
    results = {}

    # --- Original paper models ---
    results["ARIMA"] = evaluate_arima_on_workload(
        df, train_frac=train_frac, test_len=test_len, value_col=value_col)

    results["SVR"] = evaluate_svr_on_workload(
        df, train_frac=train_frac, test_len=test_len, window=window,
        value_col=value_col)

    results["RandomForest"] = evaluate_rf_on_workload(
        df, train_frac=train_frac, test_len=test_len, window=window,
        value_col=value_col)

    # --- Novelty additions ---
    if include_novelty:
        if evaluate_xgboost_on_workload is not None:
            try:
                results["XGBoost"] = evaluate_xgboost_on_workload(
                    df, train_frac=train_frac, test_len=test_len, window=window,
                    value_col=value_col)
            except Exception as e:
                print(f"[Warning] XGBoost failed ({e}), skipping.")

        if evaluate_lstm_on_workload is not None:
            try:
                results["LSTM"] = evaluate_lstm_on_workload(
                    df, train_frac=train_frac, test_len=test_len, window=window,
                    value_col=value_col)
            except Exception as e:
                print(f"[Warning] LSTM failed ({e}), skipping.")

    return results


def select_active_model(results):
    """Phase 3, step 12-13: order scores ascending, pick the lowest (best)."""
    scored = {name: r["Score"] for name, r in results.items()}
    active_model = min(scored, key=scored.get)
    return active_model, scored


def results_to_table(results):
    """Formats results as a DataFrame matching the paper's Table 5-8 layout,
    extended with MAPE column for richer comparison."""
    rows = []
    for name, r in results.items():
        row = {
            "Algorithm": name,
            "RMSE": round(r["RMSE"], 4),
            "MAE": round(r["MAE"], 4),
            "Score": f"{r['Score']:.2f}%",
        }
        if "MAPE" in r:
            row["MAPE"] = f"{r['MAPE']:.2f}%"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Algorithm")


if __name__ == "__main__":
    from synthetic_workload import PATTERNS

    for pattern_name, generator in PATTERNS.items():
        print(f"\n{'=' * 60}")
        print(f"Workload pattern: {pattern_name}")
        print("=" * 60)

        df = generator(duration_min=697)
        results = evaluate_all_models(df, include_novelty=True)

        table = results_to_table(results)
        print(table)

        active_model, scores = select_active_model(results)
        print(f"\n>>> Selected active model: {active_model} "
              f"(score={scores[active_model]:.2f}%)")
