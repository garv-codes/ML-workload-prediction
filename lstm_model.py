"""
Neural Network / LSTM workload predictor — NOVELTY addition to the MTTD framework.

The original Nashaat et al. (2026) paper uses only ARIMA, SVR, and Random
Forest. Adding neural modeling extends the candidate pool with non-linear
temporal representations.

Supports:
1. PyTorch LSTM (when torch is installed)
2. TensorFlow/Keras LSTM (when tensorflow is installed)
3. Scikit-learn MLPRegressor Neural Network (built-in zero-dependency fallback,
   ensuring full pipeline execution even if Windows MAX_PATH limits torch install)

Architecture:
- LSTM: 32 hidden units -> Linear/Dense(1)
- MLP fallback: (64, 32) hidden layers with early stopping

Exposes the same evaluate_*_on_workload() interface as the other models.
"""

import numpy as np

from sliding_window import make_supervised, train_test_split_series
from arima_baseline import rmse, mae, mape, performance_score


def _fit_predict_torch(X_train, y_train, X_test, window, units=32, epochs=60, lr=0.01):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)

    class LSTMRegressor(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if std < 1e-6:
        std = 1.0

    X_train_norm = (X_train - mean) / std
    y_train_norm = (y_train - mean) / std
    X_test_norm = (X_test - mean) / std

    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train_norm, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMRegressor(hidden_size=units)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_norm = model(X_test_t).numpy()
    preds = preds_norm * std + mean
    return preds


def _fit_predict_tf(X_train, y_train, X_test, window, units=32, epochs=50):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    X_train_3d = X_train.reshape(X_train.shape[0], window, 1)
    X_test_3d = X_test.reshape(X_test.shape[0], window, 1)

    model = Sequential([
        LSTM(units, input_shape=(window, 1), activation="tanh"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.005), loss="mse")
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X_train_3d, y_train, epochs=epochs, batch_size=16, callbacks=[es], verbose=0)
    return model.predict(X_test_3d, verbose=0).flatten()


def _fit_predict_mlp(X_train, y_train, X_test):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    X_test_s = scaler_x.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True,
    )
    mlp.fit(X_train_s, y_train_s)
    preds_s = mlp.predict(X_test_s)
    return scaler_y.inverse_transform(preds_s.reshape(-1, 1)).flatten()


def evaluate_lstm_on_workload(df, train_frac=0.7, test_len=100, window=5,
                               value_col="avg_requests_per_min",
                               units=32, epochs=50):
    """End-to-end evaluation matching the interface of SVR/RF/XGBoost evaluators."""
    backend = None
    try:
        import torch
        # Quick sanity check on torch tensor creation
        _ = torch.zeros(1)
        backend = "torch"
    except Exception:
        try:
            import tensorflow
            backend = "tf"
        except Exception:
            backend = "mlp"

    values = df[value_col].values
    train, test = train_test_split_series(values, train_frac, test_len)

    full_context = np.concatenate([train, test])
    X_all, y_all = make_supervised(full_context, window=window)

    n_train_samples = len(train) - window
    X_train, y_train = X_all[:n_train_samples], y_all[:n_train_samples]
    X_test, y_test = X_all[n_train_samples:], y_all[n_train_samples:]

    if backend == "torch":
        preds = _fit_predict_torch(X_train, y_train, X_test, window=window, units=units, epochs=epochs)
        model_name = "LSTM"
    elif backend == "tf":
        preds = _fit_predict_tf(X_train, y_train, X_test, window=window, units=units, epochs=epochs)
        model_name = "LSTM"
    else:
        import warnings as _w
        _w.warn(
            "Neither PyTorch nor TensorFlow is available; falling back to "
            "scikit-learn MLPRegressor (not a true LSTM).  The model will be "
            "labelled 'NeuralNet-MLP' in all outputs.",
            stacklevel=2,
        )
        preds = _fit_predict_mlp(X_train, y_train, X_test)
        model_name = "NeuralNet-MLP"

    return {
        "model_name": model_name,
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
        result = evaluate_lstm_on_workload(df)
        print(f"{result['model_name']} -> RMSE: {result['RMSE']:.4f}  "
              f"MAE: {result['MAE']:.4f}  "
              f"MAPE: {result['MAPE']:.2f}%  "
              f"Score: {result['Score']:.2f}%")
