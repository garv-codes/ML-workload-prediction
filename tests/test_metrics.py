"""
test_metrics.py - Unit tests for performance metrics (RMSE, MAE, MAPE, Performance Score).
"""
import numpy as np
import pytest
from arima_baseline import rmse, mae, mape, performance_score


def test_basic_rmse_mae():
    y_actual = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    # Differences: [-2, +2, -3]
    # Squared diffs: [4, 4, 9] -> mean = 17/3
    expected_rmse = np.sqrt(17.0 / 3.0)
    expected_mae = (2.0 + 2.0 + 3.0) / 3.0

    assert np.isclose(rmse(y_actual, y_pred), expected_rmse)
    assert np.isclose(mae(y_actual, y_pred), expected_mae)


def test_zero_error():
    y_actual = np.array([5.0, 10.0, 15.0])
    y_pred = np.array([5.0, 10.0, 15.0])

    assert rmse(y_actual, y_pred) == 0.0
    assert mae(y_actual, y_pred) == 0.0
    assert mape(y_actual, y_pred) == 0.0
    assert performance_score(y_actual, y_pred) == 0.0


def test_performance_score_hand_computed_eq4():
    """
    Validates Eq. 4 against hand-calculated values on a tiny example:
      y_actual = [10.0, 20.0]
      y_pred   = [12.0, 18.0]
    
    RMSE = sqrt(((-2)^2 + 2^2)/2) = 2.0
    MAE  = (2 + 2)/2 = 2.0
    sqrt(mean(y^2)) = sqrt((100 + 400)/2) = sqrt(250)
    mean(y) = 15.0
    
    Score = 100 * (0.5 * (2.0 / sqrt(250)) + 0.5 * (2.0 / 15.0))
          ~ 12.991221987...%
    """
    y_actual = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])

    expected_norm_rmse = 2.0 / np.sqrt(250.0)
    expected_norm_mae = 2.0 / 15.0
    expected_score = 100.0 * (0.5 * expected_norm_rmse + 0.5 * expected_norm_mae)

    computed_score = performance_score(y_actual, y_pred, w1=0.5, w2=0.5)
    assert np.isclose(computed_score, expected_score, rtol=1e-6)
    assert np.isclose(computed_score, 12.991221987, atol=1e-4)


def test_mape():
    y_actual = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    # |10/100| = 0.1, |10/200| = 0.05 -> mean = 0.075 -> 7.5%
    assert np.isclose(mape(y_actual, y_pred), 7.5)
