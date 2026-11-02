"""
reselection_study.py  --  Reselection-frequency experiment for the MTTD framework.

Imports mttd_bridge unchanged and runs two experiments:

  A. Standard MTTD run, keeping every candidate's hourly forecasts, then
     re-choosing the active model offline every 24 h or every 1 h by the
     lowest Score over the previous 6, 24, or 72 h, with no extra retraining.

  B. MTTD with retraining and reselection forced at the end of every 24-hour
     cycle (44 retraining events).

Provisioning outcomes use the same formulas as ProvisioningSimulation.java
(C = 4,000 requests per VM-hour, headroom 1.10, 200-request cloudlets of 180 s).

Outputs:
  - reselection_results.csv          Summary table
  - predictions_daily_retrain.csv    Predictions from Experiment B
  - reselect_hourly_24h.csv          Predictions from best-performing offline
                                     reselection (every 1 h on last 24 h)

Takes roughly 8 minutes on one CPU core, mostly the 44 retrains in Experiment B.

Usage:
    python reselection_study.py --csv data/cricket_hourly_workload.csv
"""
import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from arima_baseline import performance_score, rmse, mae
import mttd_bridge as mb

# --------------- provisioning constants (same as ProvisioningSimulation.java)
HITS_PER_VM_PER_HOUR = 4000.0
HEADROOM = 1.10
MAX_VMS = 150
SLA_SECONDS = 3600.0
HITS_PER_CLOUDLET = 200.0
VM_MIPS = 1000
MI_PER_HIT = (VM_MIPS * 3600.0) / HITS_PER_VM_PER_HOUR  # 900 MI


def vms_for(hits, headroom=HEADROOM):
    v = math.ceil(hits * headroom / HITS_PER_VM_PER_HOUR)
    return max(1, min(MAX_VMS, v))


def simple_makespan(vms, hits):
    """Approximate cloudlet makespan (seconds) matching the CloudSim model."""
    if hits <= 0:
        return 0.0
    n_cloudlets = math.ceil(hits / HITS_PER_CLOUDLET)
    length_mi = HITS_PER_CLOUDLET * MI_PER_HIT
    # Each VM does VM_MIPS MI/s; cloudlets are spread round-robin
    per_vm = math.ceil(n_cloudlets / vms)
    return per_vm * (length_mi / VM_MIPS)


def provisioning_metrics(actuals, preds):
    """Return VM-hours and SLA-violation count using MTTD-Retrain policy."""
    vm_hours = 0
    violations = 0
    for a, p in zip(actuals, preds):
        vms = vms_for(p)
        vm_hours += vms
        ms = simple_makespan(vms, a)
        if ms > SLA_SECONDS:
            violations += 1
    return int(vm_hours), int(violations)


# --------------- Experiment A: offline reselection on standard MTTD forecasts
def experiment_a(series, skip=()):
    """Run standard MTTD once, collect all candidates' forecasts, then
    re-select the active model offline at different frequencies/windows."""
    print("=== Experiment A: Standard MTTD run + offline reselection ===")
    pred_df, log_df, per_model = mb.run(series, skip=skip, allow_retrain=True)
    t0 = mb.TRAIN_LEN + mb.TEST_LEN  # 264

    # Build per-model per-hour arrays aligned to the deployment window
    dep_log = log_df[log_df.phase.str.startswith("Deploy")]
    n_hours = len(pred_df)
    model_names = sorted(per_model.keys())

    # per_model[name] = ([actuals], [preds]) aligned hour-by-hour
    actuals = np.array(per_model[model_names[0]][0])

    # Standard MTTD result
    std_preds = pred_df["predicted"].values
    std_score = performance_score(actuals, std_preds)
    std_vm, std_viol = provisioning_metrics(actuals, std_preds)
    results = [{
        "Strategy": "Standard MTTD",
        "Score": round(std_score, 2),
        "Switches": 0,
        "Models_used": 1,
        "VM_hours": std_vm,
        "SLA_violations": std_viol,
    }]

    # Offline reselection strategies
    strategies = [
        ("Reselect every 24 h on last 24 h", 24, 24),
        ("Reselect every 24 h on last 72 h", 24, 72),
        ("Reselect every 1 h on last 24 h",   1, 24),
        ("Reselect every 1 h on last 6 h",    1,  6),
    ]

    best_hourly_24_preds = None

    for strat_name, resel_every, lookback in strategies:
        # Build predictions array by choosing best model in each window
        chosen_preds = np.empty(n_hours)
        active_model = None
        switches = 0
        models_used = set()

        for h in range(n_hours):
            if h % resel_every == 0 and h >= lookback:
                # Find model with lowest Score over lookback window
                best_name, best_sc = None, float("inf")
                for name in model_names:
                    a_slice = np.array(per_model[name][0][h - lookback:h])
                    p_slice = np.array(per_model[name][1][h - lookback:h])
                    sc = performance_score(a_slice, p_slice)
                    if sc < best_sc:
                        best_sc = sc
                        best_name = name
                if active_model is not None and best_name != active_model:
                    switches += 1
                active_model = best_name
            elif active_model is None:
                # Before first reselection, use standard MTTD's active
                active_model = pred_df.iloc[h]["active_model"]

            models_used.add(active_model)
            chosen_preds[h] = per_model[active_model][1][h]

        sc = performance_score(actuals, chosen_preds)
        vm, viol = provisioning_metrics(actuals, chosen_preds)
        results.append({
            "Strategy": strat_name,
            "Score": round(sc, 2),
            "Switches": switches,
            "Models_used": len(models_used),
            "VM_hours": vm,
            "SLA_violations": viol,
        })

        if strat_name == "Reselect every 1 h on last 24 h":
            best_hourly_24_preds = chosen_preds.copy()

        print(f"  {strat_name}: Score={sc:.2f}%, switches={switches}, "
              f"models={len(models_used)}, VM-h={vm}, SLA-viol={viol}")

    return results, pred_df, per_model, best_hourly_24_preds, actuals


# --------------- Experiment B: retrain + reselect every 24 h
def experiment_b(series, skip=()):
    """MTTD with retraining and reselection forced at every 24-hour cycle."""
    print("\n=== Experiment B: Retrain + reselect every 24 h ===")
    t0 = mb.TRAIN_LEN + mb.TEST_LEN  # 264

    # Initial training
    models = mb.train_all(series[t0 - mb.TRAIN_LEN - mb.TEST_LEN:t0 - mb.TEST_LEN], skip)
    best, scores, details = mb.test_and_select(models, series, t0)
    print(f"  [t={t0}] initial: {best} (Score {scores[best]:.2f}%)")

    # Refresh ARIMA state
    for m in models:
        if m.name == "ARIMA":
            m.model.update(series[t0 - mb.TEST_LEN:t0])

    rows = []
    active_name = best
    switches = 0
    models_used = {active_name}
    retrain_count = 0
    t = t0

    while t < len(series):
        end = min(t + mb.CYCLE, len(series))
        actual_cycle = series[t:end]
        active = next(m for m in models if m.name == active_name)

        # Forecast this cycle with the active model
        preds = mb.rolling_forecast(active, series[:t], actual_cycle)
        for h, a, p in zip(range(t, end), actual_cycle, preds):
            rows.append((h, a, p, active_name, 0))

        # After every cycle: retrain + reselect
        t = end
        if t < len(series):
            retrain_count += 1
            lo = max(0, t - mb.TRAIN_LEN - mb.TEST_LEN)
            models = mb.train_all(series[lo:t - mb.TEST_LEN], skip)
            new_best, new_scores, _ = mb.test_and_select(models, series, t)
            # Refresh ARIMA
            for m in models:
                if m.name == "ARIMA":
                    m.model.update(series[t - mb.TEST_LEN:t])
            if new_best != active_name:
                switches += 1
            active_name = new_best
            models_used.add(active_name)
            rows[-1] = rows[-1][:4] + (1,)  # flag retrain
            if retrain_count % 10 == 0:
                print(f"  ... {retrain_count} retrains done (hour {t})")

    pred_df = pd.DataFrame(rows, columns=["hour", "actual", "predicted",
                                           "active_model", "retrain_event"])
    actuals = pred_df["actual"].values
    preds = pred_df["predicted"].values
    sc = performance_score(actuals, preds)
    vm, viol = provisioning_metrics(actuals, preds)

    print(f"  Retrain+reselect every 24h: Score={sc:.2f}%, switches={switches}, "
          f"models={len(models_used)}, retrains={retrain_count}, VM-h={vm}, SLA-viol={viol}")

    result = {
        "Strategy": "Retrain + reselect every 24 h",
        "Score": round(sc, 2),
        "Switches": switches,
        "Models_used": len(models_used),
        "VM_hours": vm,
        "SLA_violations": viol,
    }
    return result, pred_df


def main():
    ap = argparse.ArgumentParser(description="MTTD reselection-frequency study")
    ap.add_argument("--csv", default="data/cricket_hourly_workload.csv",
                    help="path to workload CSV")
    ap.add_argument("--outdir", default=".",
                    help="directory for output files (default: current dir)")
    ap.add_argument("--skip", default="", help="comma list from: svr,rf,arima,xgb,lstm")
    a = ap.parse_args()

    series = mb.load_series(a.csv)
    print(f"Loaded {len(series)} hourly points")
    skip = set(x.strip() for x in a.skip.split(",") if x.strip())
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Experiment A
    results_a, std_pred_df, per_model, hourly24_preds, actuals = experiment_a(series, skip)

    # Experiment B
    result_b, daily_pred_df = experiment_b(series, skip)

    # Combine results
    all_results = results_a + [result_b]
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(outdir / "reselection_results.csv", index=False)
    print(f"\nSaved reselection_results.csv ({len(res_df)} strategies)")

    # Save predictions
    daily_pred_df.to_csv(outdir / "predictions_daily_retrain.csv",
                         index=False, float_format="%.2f")
    print(f"Saved predictions_daily_retrain.csv ({len(daily_pred_df)} rows)")

    # Save hourly-24h reselection predictions
    if hourly24_preds is not None:
        t0 = mb.TRAIN_LEN + mb.TEST_LEN
        hourly_df = pd.DataFrame({
            "hour": range(t0, t0 + len(actuals)),
            "actual": actuals,
            "predicted": hourly24_preds,
        })
        hourly_df.to_csv(outdir / "reselect_hourly_24h.csv",
                         index=False, float_format="%.2f")
        print(f"Saved reselect_hourly_24h.csv ({len(hourly_df)} rows)")

    # Print summary
    print("\n=== Reselection Study Summary ===")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
