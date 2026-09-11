"""
SVR and Random Forest predictors, using the sliding-window supervised
transform + grid search for hyperparameters (as described in the paper's
'Framework implementation' section).

Both models expose the same interface as arima_baseline.py's
evaluate_arima_on_workload(), so all three can be compared side by side:
  evaluate_svr_on_workload(df, ...)   -> dict with RMSE, MAE, Score, etc.
  evaluate_rf_on_workload(df, ...)
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sliding_window import make_supervised, train_test_split_series
from arima_baseline import rmse, mae, performance_score


# ---------------------------------------------------------------------------
# SVR
# ---------------------------------------------------------------------------

SVR_PARAM_GRID = {
    "C": [1, 10],
    "epsilon": [0.01, 0.1],
    "gamma": [0.01, 0.1],
    "kernel": ["rbf"],
}


def fit_svr(X_train, y_train, param_grid=None, cv=3):
    param_grid = param_grid or SVR_PARAM_GRID
    grid = GridSearchCV(SVR(), param_grid, cv=cv, scoring="neg_mean_squared_error",
                         n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def evaluate_svr_on_workload(df, train_frac=0.7, test_len=100, window=5,
                              value_col="avg_requests_per_min"):
    values = df[value_col].values
    train, test = train_test_split_series(values, train_frac, test_len)

    # sliding window needs training data to include the tail of `train`
    # so the first test predictions have real history to look back on
    full_train_context = np.concatenate([train, test])
    X_all, y_all = make_supervised(full_train_context, window=window)

    n_train_samples = len(train) - window
    X_train, y_train = X_all[:n_train_samples], y_all[:n_train_samples]
    X_test, y_test = X_all[n_train_samples:], y_all[n_train_samples:]

    model = fit_svr(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "best_params": model.get_params(),
        "predictions": preds,
        "actual": y_test,
        "RMSE": rmse(y_test, preds),
        "MAE": mae(y_test, preds),
        "Score": performance_score(y_test, preds),
    }


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

RF_PARAM_GRID = {
    "n_estimators": [100],
    "max_depth": [10, 40],
    "min_samples_leaf": [1, 3],
    "min_samples_split": [2, 8],
}


def fit_rf(X_train, y_train, param_grid=None, cv=3):
    param_grid = param_grid or RF_PARAM_GRID
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid,
                         cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def evaluate_rf_on_workload(df, train_frac=0.7, test_len=100, window=5,
                             value_col="avg_requests_per_min"):
    values = df[value_col].values
    train, test = train_test_split_series(values, train_frac, test_len)

    full_train_context = np.concatenate([train, test])
    X_all, y_all = make_supervised(full_train_context, window=window)

    n_train_samples = len(train) - window
    X_train, y_train = X_all[:n_train_samples], y_all[:n_train_samples]
    X_test, y_test = X_all[n_train_samples:], y_all[n_train_samples:]

    model = fit_rf(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "best_params": model.get_params(),
        "predictions": preds,
        "actual": y_test,
        "RMSE": rmse(y_test, preds),
        "MAE": mae(y_test, preds),
        "Score": performance_score(y_test, preds),
    }


if __name__ == "__main__":
    from synthetic_workload import generate_uniform_workload, generate_random_workload

    print("=== Uniform workload ===")
    df_uniform = generate_uniform_workload(duration_min=697)
    r_svr = evaluate_svr_on_workload(df_uniform)
    r_rf = evaluate_rf_on_workload(df_uniform)
    print(f"SVR -> RMSE: {r_svr['RMSE']:.4f}  MAE: {r_svr['MAE']:.4f}  Score: {r_svr['Score']:.2f}%")
    print(f"RF  -> RMSE: {r_rf['RMSE']:.4f}  MAE: {r_rf['MAE']:.4f}  Score: {r_rf['Score']:.2f}%")

    print("\n=== Random workload ===")
    df_random = generate_random_workload(duration_min=697)
    r_svr2 = evaluate_svr_on_workload(df_random)
    r_rf2 = evaluate_rf_on_workload(df_random)
    print(f"SVR -> RMSE: {r_svr2['RMSE']:.4f}  MAE: {r_svr2['MAE']:.4f}  Score: {r_svr2['Score']:.2f}%")
    print(f"RF  -> RMSE: {r_rf2['RMSE']:.4f}  MAE: {r_rf2['MAE']:.4f}  Score: {r_rf2['Score']:.2f}%")
    print()
    print("(Paper finds SVR wins on random workloads, Table 8 -- our synthetic")
    print(" random walk should show SVR/RF closer together and both clearly")
    print(" worse than on the uniform workload, since random is harder to predict.)")
