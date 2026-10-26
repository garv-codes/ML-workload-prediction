"""
mttd_bridge.py  --  Runs the MTTD loop (Monitor-Train-Test-Deploy) on a
semi-synthetic hourly workload trace (real daily totals, synthetic hourly profile)
and exports predictions.csv for the CloudSim simulation.

Put this file in the same folder as deployment_manager.py / arima_baseline.py.

Usage:
    python mttd_bridge.py --csv data/cricket_hourly_workload.csv --out predictions.csv
    python mttd_bridge.py --csv data/cricket_hourly_workload.csv --no-arima     # if pmdarima won't install
    python mttd_bridge.py --csv data/cricket_hourly_workload.csv --no-retrain   # "MTTD without retraining" run

Output columns:
    hour, actual, predicted, active_model, retrain_event
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from deployment_manager import DeploymentManager      # Eq. 5-6, two-strike logic
from arima_baseline import performance_score, rmse, mae   # Eq. 2, 3, 4

WINDOW = 24       # hourly data -> use last 24 h as features (daily pattern)
TRAIN_LEN = 240   # 10 days, same as paper's FIFA-2018 experiment
TEST_LEN = 24     # 1 day test window (paper: 24 h for hourly data)
CYCLE = 24        # deployment cycle = 1 day


# ---------------------------------------------------------------- data
def load_series(path):
    df = pd.read_csv(path)
    df = df.iloc[:, :2]
    df.columns = ["t", "hits"]
    df["hits"] = pd.to_numeric(df["hits"], errors="coerce")
    df = df.dropna().reset_index(drop=True)      # removes the '** Process exited **' line
    return df["hits"].values.astype(float)


def make_xy(series, window=WINDOW):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------- models
class WindowModel:
    """SVR / RF on sliding window. Values are scaled by a constant so SVR behaves."""
    def __init__(self, name, use_grid=True):
        self.name, self.use_grid = name, use_grid

    def fit(self, series):
        self.scale = max(series.max(), 1.0)
        X, y = make_xy(series / self.scale)
        if self.name == "SVR":
            grid = {"C": [1, 10, 100], "epsilon": [0.001, 0.01], "gamma": [0.01, 0.1], "kernel": ["rbf"]}
            base = SVR()
        elif self.name == "XGBoost":
            from xgboost import XGBRegressor            # pip install xgboost
            grid = {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]}
            base = XGBRegressor(random_state=42, verbosity=0)
        else:
            grid = {"n_estimators": [100], "max_depth": [10, 40], "min_samples_leaf": [1, 3]}
            base = RandomForestRegressor(random_state=42)
        g = GridSearchCV(base, grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
        g.fit(X, y)
        self.model = g.best_estimator_
        return self

    def predict_next(self, history):
        x = (np.asarray(history[-WINDOW:]) / self.scale).reshape(1, -1)
        return float(self.model.predict(x)[0] * self.scale)

    def observe(self, actual):
        pass  # stateless: history is supplied by caller


class ArimaModel:
    name = "ARIMA"

    def fit(self, series):
        import pmdarima as pm
        self.model = pm.auto_arima(series, start_p=0, start_q=0, max_p=5, max_q=5, d=None,
                                   seasonal=False, stepwise=True, suppress_warnings=True,
                                   error_action="ignore")
        return self

    def predict_next(self, history):
        return float(self.model.predict(n_periods=1)[0])

    def observe(self, actual):
        self.model.update([actual])


class LSTMModel:
    """PyTorch LSTM (32 units). Falls back to an sklearn MLP if torch cannot be imported
    (e.g. Windows long-path problems) so the pipeline still runs. The fallback is renamed so
    reports stay honest."""
    name = "LSTM"

    def fit(self, series):
        self.mean, self.std = float(series.mean()), max(float(series.std()), 1.0)
        X, y = make_xy((series - self.mean) / self.std)
        try:
            import torch
            import torch.nn as nn
        except Exception:
            torch = None
        if torch is not None:
            torch.manual_seed(42)

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(1, 32, batch_first=True)
                    self.fc = nn.Linear(32, 1)

                def forward(self, x):
                    o, _ = self.lstm(x)
                    return self.fc(o[:, -1, :]).squeeze(-1)

            net = Net()
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            lossf = nn.MSELoss()
            Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
            yt = torch.tensor(y, dtype=torch.float32)
            for _ in range(60):
                perm = torch.randperm(len(Xt))
                for i in range(0, len(Xt), 32):
                    idx = perm[i:i + 32]
                    opt.zero_grad()
                    lossf(net(Xt[idx]), yt[idx]).backward()
                    opt.step()
            net.eval()
            self.net, self.backend, self.torch = net, "torch", torch
        else:
            from sklearn.neural_network import MLPRegressor
            self.net = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                    early_stopping=True, random_state=42).fit(X, y)
            self.backend, self.name = "mlp", "NeuralNet-MLP"
            print("[warn] torch not available -> using MLP fallback instead of LSTM")
        return self

    def predict_next(self, history):
        x = (np.asarray(history[-WINDOW:], dtype=float) - self.mean) / self.std
        if self.backend == "torch":
            t = self.torch.tensor(x, dtype=self.torch.float32).reshape(1, WINDOW, 1)
            with self.torch.no_grad():
                p = float(self.net(t).item())
        else:
            p = float(self.net.predict(x.reshape(1, -1))[0])
        return p * self.std + self.mean

    def observe(self, actual):
        pass


def train_all(train_series, skip=()):
    """Train every candidate. A model that fails to train/import is skipped with a warning."""
    builders = [
        ("svr", lambda: WindowModel("SVR").fit(train_series)),
        ("rf", lambda: WindowModel("RandomForest").fit(train_series)),
        ("arima", lambda: ArimaModel().fit(train_series)),
        ("xgb", lambda: WindowModel("XGBoost").fit(train_series)),
        ("lstm", lambda: LSTMModel().fit(train_series)),
    ]
    models = []
    for key, build in builders:
        if key in skip:
            continue
        try:
            models.append(build())
        except Exception as e:
            print(f"[warn] {key} skipped: {e}")
    return models


def rolling_forecast(model, history, actuals):
    """One-step-ahead over `actuals`, using real observations as they arrive."""
    hist, preds = list(history), []
    for a in actuals:
        preds.append(max(model.predict_next(hist), 0.0))
        model.observe(a)
        hist.append(a)
    return np.array(preds)


def test_and_select(models, series, t_now):
    """Phase 2+3: test every model on the last TEST_LEN hours before t_now; lowest Score wins.
    Returns (best_name, {name: Score}, {name: (RMSE, MAE, Score)})."""
    import copy
    ctx_start = t_now - TEST_LEN - WINDOW
    test_actual = series[t_now - TEST_LEN:t_now]
    scores, details = {}, {}
    for m in models:
        if m.name == "ARIMA":
            # ARIMA was fit on data ending at t_now-TEST_LEN, so forecast the test window rolling
            p = rolling_forecast(copy.deepcopy(m), series[:t_now - TEST_LEN], test_actual)
        else:
            p = rolling_forecast(m, series[ctx_start:t_now - TEST_LEN], test_actual)
        sc = performance_score(test_actual, p)
        scores[m.name] = sc
        details[m.name] = (rmse(test_actual, p), mae(test_actual, p), sc)
    best = min(scores, key=scores.get)
    return best, scores, details


# ---------------------------------------------------------------- main loop
def run(series, skip=(), allow_retrain=True):
    """Returns (predictions_df, model_log_df). model_log_df has one row per model per phase,
    like the paper's Tables 5-13 (Test phase, Deployment phase I, II, ...)."""
    log = []

    def log_phase(phase, t, details, active_name, thr=None, retrain=False):
        for name, (r, m_, sc) in details.items():
            log.append(dict(phase=phase, start_hour=t, model=name, RMSE=r, MAE=m_, Score=sc,
                            is_active=int(name == active_name), threshold=thr, retrain=int(retrain)))

    t0 = TRAIN_LEN + TEST_LEN
    models = train_all(series[t0 - TRAIN_LEN - TEST_LEN:t0 - TEST_LEN], skip)
    best, scores, details = test_and_select(models, series, t0)
    print(f"[t={t0}] models compared: {[m.name for m in models]}")
    print(f"[t={t0}] initial scores {({k: round(float(v), 2) for k, v in scores.items()})} -> active = {best}")
    log_phase("Test", t0, details, best)
    mgr = DeploymentManager(best, initial_score=scores[best])
    active = next(m for m in models if m.name == best)

    def refresh_state(models_, upto):
        for m in models_:
            if m.name == "ARIMA":
                m.model.update(series[upto - TEST_LEN:upto])
    refresh_state(models, t0)

    rows, t, cyc = [], t0, 0
    per_model = {}   # name -> ([actuals], [preds]) over the whole deployment
    while t < len(series):
        end = min(t + CYCLE, len(series))
        actual_cycle = series[t:end]
        cyc += 1
        # every candidate forecasts the cycle (paper's tables report all algorithms per phase);
        # only the ACTIVE model's forecasts are used for provisioning
        cyc_preds = {m.name: rolling_forecast(m, series[:t], actual_cycle) for m in models}
        for n_, p_ in cyc_preds.items():
            per_model.setdefault(n_, ([], []))
            per_model[n_][0].extend(actual_cycle)
            per_model[n_][1].extend(p_)
        preds = cyc_preds[active.name]
        res = mgr.run_cycle(actual_cycle, preds)
        retrain = bool(res["retrain_triggered"]) and allow_retrain
        det = {n: (rmse(actual_cycle, p), mae(actual_cycle, p), performance_score(actual_cycle, p))
               for n, p in cyc_preds.items()}
        log_phase(f"Deploy{cyc}", t, det, active.name, res["threshold_used"], retrain)
        for h, a, p in zip(range(t, end), actual_cycle, preds):
            rows.append((h, a, p, active.name, 0))
        print(f"  cycle {cyc:2d} hours {t}-{end - 1}: active={active.name:12s} score={res['score']:.2f}% "
              f"thr={res['threshold_used']:.2f}% strikes={res['strikes']} retrain={retrain}")
        t = end
        if retrain and t < len(series):
            lo = max(0, t - TRAIN_LEN - TEST_LEN)
            models = train_all(series[lo:t - TEST_LEN], skip)
            best, scores, details = test_and_select(models, series, t)
            refresh_state(models, t)
            active = next(m for m in models if m.name == best)
            mgr.set_active_model(best, scores[best])
            rows[-1] = rows[-1][:4] + (1,)       # flag the last hour of the cycle as the retrain event
            log_phase(f"Retest@{t}", t, details, best)
            print(f"  >>> RETRAIN at hour {t}: scores {({k: round(float(v), 2) for k, v in scores.items()})} -> active = {best}")
    pred_df = pd.DataFrame(rows, columns=["hour", "actual", "predicted", "active_model", "retrain_event"])
    return pred_df, pd.DataFrame(log), per_model


def summarize(log, pred_df, per_model, prefix):
    """Prints/saves the tables that answer: which model is best on THIS dataset?"""
    dep = log[log.phase.str.startswith("Deploy")]
    summ = dep.groupby("model").agg(mean_Score=("Score", "mean"), mean_RMSE=("RMSE", "mean"),
                                    mean_MAE=("MAE", "mean"), cycles=("Score", "size")).round(2)
    best_cnt = dep.loc[dep.groupby("start_hour")["Score"].idxmin()].model.value_counts()
    summ["cycles_best"] = best_cnt.reindex(summ.index).fillna(0).astype(int)
    summ["overall_Score"] = pd.Series({n: performance_score(np.array(a), np.array(p))
                                       for n, (a, p) in per_model.items()}).round(2)
    summ = summ.sort_values("overall_Score")
    used = pred_df.groupby("active_model").size().rename("hours_active")
    summ = summ.join(used).fillna({"hours_active": 0}).astype({"hours_active": int})
    print("\n=== Model comparison over all deployment cycles (lower Score = better) ===")
    print(summ.to_string())
    mttd_overall = performance_score(pred_df.actual.values, pred_df.predicted.values)
    print(f"\nBest single model on this dataset (lowest overall_Score): {summ.index[0]} ({summ.overall_Score.iloc[0]:.2f}%)")
    print(f"MTTD (dynamic switching) overall Score: {mttd_overall:.2f}%")
    print("(all models above are retrained at the same MTTD retraining events; "
          "overall_Score = Eq. 4 on the full deployment period)")
    log.round(3).to_csv(prefix + "_model_scores.csv", index=False)
    summ.to_csv(prefix + "_model_summary.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for name, g in dep.groupby("model"):
            ax[0].plot(g.start_hour, g.Score, marker="o", ms=3, label=name)
        ax[0].set_ylabel("Score % (lower = better)")
        ax[0].set_yscale("log")
        ax[0].legend(ncol=3, fontsize=8)
        ax[0].set_title("Per-cycle Score of every candidate model (deployment phase)")
        cyc = dep[dep.is_active == 1].drop_duplicates("start_hour")
        names = sorted(dep.model.unique())
        ax[1].step(cyc.start_hour, [names.index(n) for n in cyc.model], where="post")
        ax[1].set_yticks(range(len(names)))
        ax[1].set_yticklabels(names)
        ax[1].set_title("Active model selected by MTTD over time")
        ax[1].set_xlabel("hour")
        for h in pred_df[pred_df.retrain_event == 1].hour:
            for a_ in ax:
                a_.axvline(h, color="red", ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(prefix + "_model_comparison.png", dpi=130)
        print(f"Saved {prefix}_model_scores.csv, {prefix}_model_summary.csv, {prefix}_model_comparison.png")
    except Exception as e:
        print(f"[warn] chart skipped: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/cricket_hourly_workload.csv",
                    help="path to workload CSV (default: data/cricket_hourly_workload.csv)")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--no-arima", action="store_true")
    ap.add_argument("--skip", default="", help="comma list from: svr,rf,arima,xgb,lstm")
    ap.add_argument("--no-retrain", action="store_true")
    a = ap.parse_args()
    s = load_series(a.csv)
    print(f"Loaded {len(s)} hourly points, min={s.min():.0f} max={s.max():.0f}")
    skip = set(x.strip() for x in a.skip.split(",") if x.strip())
    if a.no_arima:
        skip.add("arima")
    out, log, per_model = run(s, skip=skip, allow_retrain=not a.no_retrain)
    out.to_csv(a.out, index=False, float_format="%.2f")
    summarize(log, out, per_model, a.out.rsplit(".", 1)[0])
    sc = performance_score(out["actual"].values, out["predicted"].values)
    print(f"\nWrote {a.out} ({len(out)} rows). Overall Score = {sc:.2f}%")
