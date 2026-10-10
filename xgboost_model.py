"""
XGBoost workload predictor — NOVELTY addition to the MTTD framework.

XGBoost (eXtreme Gradient Boosting) is a powerful gradient-boosted tree
algorithm that often outperforms Random Forest on tabular/structured data.
Adding it to the candidate pool tests whether boosting-based methods
provide better workload predictions than the paper's bagging-based RF.

Uses the same sliding-window supervised transform and grid-search tuning
as the SVR/RF models to ensure a fair comparison.

Exposes the same evaluate_*_on_workload() interface as the other models.
"""

import numpy as np
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
from sklearn.model_selection import GridSearchCV

from sliding_window import make_supervised, train_test_split_series
from arima_baseline import rmse, mae, mape, performance_score


XGBOOST_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}


def fit_xgboost(X_train, y_train, param_grid=None, cv=3):
    """Grid-search tuned XGBoost regressor, same protocol as SVR/RF."""
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    param_grid = param_grid or XGBOOST_PARAM_GRID
    grid = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def evaluate_xgboost_on_workload(df, train_frac=0.7, test_len=100, window=5,
                                  value_col="avg_requests_per_min"):
    """End-to-end evaluation matching the interface of SVR/RF/LSTM evaluators."""
    values = df[value_col].values
    train, test = train_test_split_series(values, train_frac, test_len)

    full_context = np.concatenate([train, test])
    X_all, y_all = make_supervised(full_context, window=window)

    n_train_samples = len(train) - window
    X_train, y_train = X_all[:n_train_samples], y_all[:n_train_samples]
    X_test, y_test = X_all[n_train_samples:], y_all[n_train_samples:]

    model = fit_xgboost(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model_name": "XGBoost",
        "best_params": model.get_params(),
        "predictions": preds,
        "actual": y_test,
        "RMSE": rmse(y_test, preds),
        "MAE": mae(y_test, preds),
        "MAPE": mape(y_test, preds),
        "Score": performance_score(y_test, preds),
    }


if __name__ == "__main__":
    from synthetic_workload import PATTERNS

    for name, gen in PATTERNS.items():
        print(f"\n=== {name} workload ===")
        df = gen(duration_min=697)
        result = evaluate_xgboost_on_workload(df)
        print(f"XGBoost -> RMSE: {result['RMSE']:.4f}  "
              f"MAE: {result['MAE']:.4f}  "
              f"MAPE: {result['MAPE']:.2f}%  "
              f"Score: {result['Score']:.2f}%")
