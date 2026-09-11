"""
Wraps a fitted ARIMA / SVR / RandomForest model behind one common interface
so the deployment loop (Phase 4) doesn't need to know which algorithm is
active. Every predictor supports:

    predict_next() -> float          # forecast the next minute
    observe(actual_value)            # feed in the real observed value
                                      # (updates internal state/history)

This mirrors the paper's description: "Run ML_A and predict HTTP requests
for the next time unit based on previous l time units" (Fig. 3, step 18).
"""

import numpy as np


class ArimaPredictor:
    """Wraps a fitted pmdarima model. ARIMA models natively support
    .update() with new observations, so no manual history buffer needed."""

    name = "ARIMA"

    def __init__(self, fitted_model):
        self.model = fitted_model

    def predict_next(self):
        return float(self.model.predict(n_periods=1)[0])

    def observe(self, actual_value):
        self.model.update([actual_value])


class WindowModelPredictor:
    """Wraps a fitted sklearn regressor (SVR or RandomForest) that expects
    sliding-window features. Keeps a rolling history buffer of the last
    `window` observed values to build the next feature vector."""

    def __init__(self, fitted_model, window, initial_history, name):
        self.model = fitted_model
        self.window = window
        self.history = list(initial_history[-window:])
        self.name = name

    def predict_next(self):
        x = np.array(self.history[-self.window:]).reshape(1, -1)
        return float(self.model.predict(x)[0])

    def observe(self, actual_value):
        self.history.append(actual_value)


def make_svr_predictor(fitted_svr, window, initial_history):
    return WindowModelPredictor(fitted_svr, window, initial_history, name="SVR")


def make_rf_predictor(fitted_rf, window, initial_history):
    return WindowModelPredictor(fitted_rf, window, initial_history, name="RandomForest")
