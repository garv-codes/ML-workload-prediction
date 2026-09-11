"""
Model selection logic -- Phase 3 ("Testing and Model Selection") of the MTTD
framework, following Fig. 3's pseudocode (steps 5-13):

  For each candidate model:
      train it on the training window
      test it on the test window
      compute its Performance Score
  Select the model with the LOWEST score as the active model.

This module ties together arima_baseline.py and svr_rf_models.py and
produces the same kind of results table the paper shows (Tables 5-8).
"""

import pandas as pd

from arima_baseline import evaluate_arima_on_workload
from svr_rf_models import evaluate_svr_on_workload, evaluate_rf_on_workload


def evaluate_all_models(df, train_frac=0.7, test_len=100, window=5,
                         value_col="avg_requests_per_min"):
    """Runs ARIMA, SVR, and Random Forest on the same workload/test split
    and returns a dict of results keyed by model name."""
    results = {}

    results["ARIMA"] = evaluate_arima_on_workload(
        df, train_frac=train_frac, test_len=test_len, value_col=value_col)

    results["SVR"] = evaluate_svr_on_workload(
        df, train_frac=train_frac, test_len=test_len, window=window,
        value_col=value_col)

    results["RandomForest"] = evaluate_rf_on_workload(
        df, train_frac=train_frac, test_len=test_len, window=window,
        value_col=value_col)

    return results


def select_active_model(results):
    """Phase 3, step 12-13: order scores ascending, pick the lowest (best)."""
    scored = {name: r["Score"] for name, r in results.items()}
    active_model = min(scored, key=scored.get)
    return active_model, scored


def results_to_table(results):
    """Formats results as a DataFrame matching the paper's Table 5-8 layout."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Algorithm": name,
            "RMSE": round(r["RMSE"], 4),
            "MAE": round(r["MAE"], 4),
            "Score": f"{r['Score']:.2f}%",
        })
    return pd.DataFrame(rows).set_index("Algorithm")


if __name__ == "__main__":
    from synthetic_workload import PATTERNS

    for pattern_name, generator in PATTERNS.items():
        print(f"\n{'=' * 50}")
        print(f"Workload pattern: {pattern_name}")
        print("=" * 50)

        df = generator(duration_min=697)
        results = evaluate_all_models(df)

        table = results_to_table(results)
        print(table)

        active_model, scores = select_active_model(results)
        print(f"\n>>> Selected active model: {active_model} "
              f"(score={scores[active_model]:.2f}%)")
