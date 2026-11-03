"""
test_deployment_manager.py - Unit tests for Phase 4 Continuous Deployment state machine.

Tests:
  1. Healthy cycle resets strikes.
  2. One breach gives strike 1 and applies Eq. 6 threshold.
  3. Two consecutive breaches trigger retraining.
  4. set_active_model resets state.
  5. A single bad cycle never triggers retraining.
"""
import numpy as np
import pytest
from deployment_manager import DeploymentManager, L1_DEFAULT, L2_DEFAULT


def make_dummy_data(target_score):
    """
    Returns actual and predicted arrays that yield approximately target_score
    using performance_score formula.
    """
    # For actual = [100.0], predicted = [100.0 + err]:
    # RMSE = err, MAE = err, norm_rmse_denom = 100, norm_mae_denom = 100
    # Score = 100 * (0.5 * err/100 + 0.5 * err/100) = err
    actual = np.full(20, 100.0)
    pred = np.full(20, 100.0 + target_score)
    return actual, pred


def test_initial_threshold():
    initial_score = 10.0
    dm = DeploymentManager(active_model_name="ARIMA", initial_score=initial_score, l1=0.75, l2=0.50)
    # Eq. 5: Score_0 + 0.75 * Score_0 = 17.5
    assert dm.state.threshold == 17.5
    assert dm.state.strikes == 0
    assert dm.state.active_model_name == "ARIMA"


def test_healthy_cycle_resets_strikes_and_refreshes_threshold():
    dm = DeploymentManager(active_model_name="ARIMA", initial_score=10.0, l1=0.75, l2=0.50)
    # Run healthy cycle (score 8.0 <= 17.5)
    actual, pred = make_dummy_data(8.0)
    res = dm.run_cycle(actual, pred)

    assert not res["retrain_triggered"]
    assert res["strikes"] == 0
    assert np.isclose(res["score"], 8.0)
    # Refreshed threshold from healthy cycle: 8.0 + 0.75 * 8.0 = 14.0
    assert np.isclose(dm.state.threshold, 14.0)


def test_single_bad_cycle_never_triggers_retrain():
    dm = DeploymentManager(active_model_name="ARIMA", initial_score=10.0, l1=0.75, l2=0.50)
    # Initial threshold = 17.5
    # Trigger a breach (score 20.0 > 17.5)
    actual, pred = make_dummy_data(20.0)
    res = dm.run_cycle(actual, pred)

    assert not res["retrain_triggered"]
    assert res["strikes"] == 1
    # Eq. 6: Score_t + 0.50 * Score_t = 20.0 + 0.50 * 20.0 = 30.0
    assert np.isclose(dm.state.threshold, 30.0)


def test_healthy_cycle_after_strike_resets_strike():
    dm = DeploymentManager(active_model_name="ARIMA", initial_score=10.0, l1=0.75, l2=0.50)
    # Cycle 1: breach -> strike 1 (threshold becomes 30.0)
    actual, pred = make_dummy_data(20.0)
    res1 = dm.run_cycle(actual, pred)
    assert res1["strikes"] == 1
    assert not res1["retrain_triggered"]

    # Cycle 2: healthy under new threshold (score 15.0 <= 30.0)
    actual2, pred2 = make_dummy_data(15.0)
    res2 = dm.run_cycle(actual2, pred2)
    assert res2["strikes"] == 0
    assert not res2["retrain_triggered"]
    # Threshold refreshed from healthy cycle: 15.0 + 0.75 * 15.0 = 26.25
    assert np.isclose(dm.state.threshold, 26.25)


def test_two_consecutive_breaches_trigger_retraining():
    dm = DeploymentManager(active_model_name="ARIMA", initial_score=10.0, l1=0.75, l2=0.50)
    # Cycle 1: breach (score 20.0 > 17.5) -> strike 1, threshold becomes 30.0
    actual1, pred1 = make_dummy_data(20.0)
    res1 = dm.run_cycle(actual1, pred1)
    assert res1["strikes"] == 1
    assert not res1["retrain_triggered"]

    # Cycle 2: consecutive breach (score 35.0 > 30.0) -> strike 2 -> retrain triggered!
    actual2, pred2 = make_dummy_data(35.0)
    res2 = dm.run_cycle(actual2, pred2)
    assert res2["retrain_triggered"]
    assert dm.state.strikes == 0  # resets after triggering retraining


def test_set_active_model_resets_state():
    dm = DeploymentManager(active_model_name="ARIMA", initial_score=10.0, l1=0.75, l2=0.50)
    # Accumulate a strike
    actual1, pred1 = make_dummy_data(20.0)
    dm.run_cycle(actual1, pred1)
    assert dm.state.strikes == 1

    # Switch active model
    dm.set_active_model("RandomForest", fresh_score=6.0)
    assert dm.state.active_model_name == "RandomForest"
    assert dm.state.strikes == 0
    # Threshold reset to fresh_score * (1 + l1) = 6.0 * 1.75 = 10.5
    assert np.isclose(dm.state.threshold, 10.5)
