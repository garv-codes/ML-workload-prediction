"""
Sliding-window transform: converts a raw time series into a supervised
learning table, as described in the paper for SVR and Random Forest:

  "SVR ... needs to transform the dataset into a supervised learning problem
   using a sliding window of n minutes, where each record contains the
   previous n minutes of workloads and outputs the predicted workload for
   the next minute."

Example with window=3 on series [10, 12, 15, 20, 18]:
  X = [[10, 12, 15],     y = [20,
       [12, 15, 20]]          18]
"""

import numpy as np


def make_supervised(series, window=5):
    """series: 1D array-like of workload values.
    window: number of past minutes used as features.
    Returns (X, y) as numpy arrays, X.shape = (n_samples, window).
    """
    series = np.asarray(series, dtype=float)
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i: i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def train_test_split_series(values, train_frac=0.7, test_len=100):
    """Same split convention used in arima_baseline.py, kept consistent
    across all models so results are comparable."""
    train_end = int(len(values) * train_frac)
    train = values[:train_end]
    test = values[train_end: train_end + test_len]
    return train, test
