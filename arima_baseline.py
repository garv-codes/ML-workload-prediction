"""
ARIMA baseline predictor + the paper's evaluation metrics.

Implements:
  - fit_arima(): trains ARIMA with auto-selected (p,d,q) via pmdarima's AutoARIMA
                 (mirrors the paper's PACF/ACF-based parameter selection, Sec 4.3)
  - forecast_arima(): one-step-ahead rolling forecast over a test window
  - rmse(), mae(): Eq. 2 and Eq. 3
  - performance_score(): Eq. 4, the weighted, normalized RMSE+MAE score
"""

import numpy as np
import pandas as pd
import pmdarima as pm


# ---------------------------------------------------------------------------
# Metrics (Eq. 2, 3, 4 in the paper)
# ---------------------------------------------------------------------------

def rmse(y_actual, y_pred):
    y_actual, y_pred = np.asarray(y_actual), np.asarray(y_pred)
    return np.sqrt(np.mean((y_actual - y_pred) ** 2))


def mae(y_actual, y_pred):
    y_actual, y_pred = np.asarray(y_actual), np.asarray(y_pred)
    return np.mean(np.abs(y_actual - y_pred))


def performance_score(y_actual, y_pred, w1=0.5, w2=0.5):
    """Eq. 4: weighted, normalized combination of RMSE and MAE.
    Lower is better. ~0% = near-perfect prediction.
    w1, w2 default to 0.5/0.5 as used throughout the paper's experiments.
    """
    y_actual = np.asarray(y_actual)
    r = rmse(y_actual, y_pred)
    m = mae(y_actual, y_pred)

    norm_rmse_denom = np.sqrt(np.mean(y_actual ** 2))
    norm_mae_denom = np.mean(y_actual)

    score = (w1 * (r / norm_rmse_denom) + w2 * (m / norm_mae_denom)) * 100
    return score


# ---------------------------------------------------------------------------
# ARIMA model
# ---------------------------------------------------------------------------

def fit_arima(train_series):
    """Fits an ARIMA model with (p,d,q) chosen automatically, analogous to
    the paper's AutoARIMA + ACF/PACF tuning (Sec. 'Framework implementation').
    train_series: 1D array-like of workload values (training window only).
    """
    model = pm.auto_arima(
        train_series,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None,             # let AutoARIMA determine differencing order
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    return model


def forecast_arima_rolling(model, train_series, test_series):
    """One-step-ahead rolling forecast: predict the next point, then update
    the model with the true observed value before predicting the next one.
    This matches how ARIMA is used operationally in the paper (predicting
    the next minute's workload from recent history)."""
    history = list(train_series)
    preds = []
    m = model  # pmdarima models support .update()
    for actual_value in test_series:
        pred = m.predict(n_periods=1)[0]
        preds.append(pred)
        m.update([actual_value])
        history.append(actual_value)
    return np.array(preds)


def evaluate_arima_on_workload(df, train_frac=0.7, test_len=100,
                                value_col="avg_requests_per_min"):
    """End-to-end: split df into train/test like the paper's Test phase,
    fit ARIMA, forecast, and compute RMSE/MAE/Score.
    Returns a dict with predictions and metrics -- same shape as Table 5's
    'Test phase' row.
    """
    values = df[value_col].values
    train_end = int(len(values) * train_frac)
    train = values[:train_end]
    test = values[train_end: train_end + test_len]

    model = fit_arima(train)
    preds = forecast_arima_rolling(model, train, test)

    return {
        "model_order": model.order,
        "predictions": preds,
        "actual": test,
        "RMSE": rmse(test, preds),
        "MAE": mae(test, preds),
        "Score": performance_score(test, preds),
    }


if __name__ == "__main__":
    from synthetic_workload import generate_uniform_workload

    df = generate_uniform_workload(duration_min=697)
    result = evaluate_arima_on_workload(df, train_frac=0.7, test_len=100)

    print(f"ARIMA order (p,d,q): {result['model_order']}")
    print(f"RMSE:  {result['RMSE']:.4f}")
    print(f"MAE:   {result['MAE']:.4f}")
    print(f"Score: {result['Score']:.2f}%")
    print()
    print("(Compare against paper Table 5, ARIMA test phase: "
          "RMSE=1.36, MAE=0.67, Score=0.47% -- exact numbers won't match "
          "since we use synthetic data, but the score should be low/stable "
          "for a smooth uniform workload, same as the paper finds.)")
