"""
run_benchmark.py — Full reproduction of Tables 5-8 from Nashaat et al. (2026)
plus extended comparison with novelty models (LSTM, XGBoost).

Outputs:
  1. Console: formatted tables for each workload pattern
  2. results/ directory:
     - CSV files for each pattern's results table
     - Prediction vs actual plots (paper Figs. 5-15 equivalent)
     - Bar chart comparing all models across patterns
     - Summary table across all patterns

Usage:
    python run_benchmark.py
    python run_benchmark.py --original-only   # only ARIMA/SVR/RF (paper reproduction)
"""

import os
import sys

# Configure UTF-8 for Windows console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from synthetic_workload import PATTERNS
from model_selector import evaluate_all_models, select_active_model, results_to_table

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DURATION_MIN = 697          # same as paper
TRAIN_FRAC = 0.7            # 70/30 train-test split (paper Sec. 4)
TEST_LEN = 100              # test window length
WINDOW = 5                  # sliding window for SVR/RF/XGBoost/LSTM
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Plot styling
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})
COLORS = {
    "ARIMA": "#2196F3",
    "SVR": "#FF9800",
    "RandomForest": "#4CAF50",
    "XGBoost": "#9C27B0",
    "LSTM": "#E91E63",
}


def ensure_results_dir():
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)


def plot_workload_patterns():
    """Reproduce Fig. 1 from the paper — all 4 workload patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = {
        "uniform": "(a) Uniform / Bell-shaped",
        "linear_low": "(b) Linear, low rate (0.35 req/s/min)",
        "linear_high": "(c) Linear, high rate (2.31 req/s/min)",
        "random": "(d) Random workload",
    }
    for ax, (name, gen) in zip(axes.flat, PATTERNS.items()):
        df = gen(duration_min=DURATION_MIN)
        ax.plot(df["minute"], df["avg_requests_per_min"],
                color="#1976D2", linewidth=0.8)
        ax.set_title(titles.get(name, name), fontweight="bold")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Requests/min")
    plt.suptitle("Figure 1: Workload Patterns (Paper Reproduction)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "figures", "fig1_workload_patterns.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_predictions(pattern_name, results):
    """Plot actual vs predicted for each model — equivalent to paper Figs. 5-15."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, r) in zip(axes, results.items()):
        actual = r["actual"]
        preds = r["predictions"]
        x = np.arange(len(actual))
        ax.plot(x, actual, label="Actual", color="#333", linewidth=1.2)
        ax.plot(x, preds, label="Predicted",
                color=COLORS.get(model_name, "#999"), linewidth=1.2,
                linestyle="--")
        ax.set_title(f"{model_name}\nScore={r['Score']:.2f}%", fontweight="bold")
        ax.set_xlabel("Test timestep")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Requests/min")

    plt.suptitle(f"Predictions: {pattern_name} workload",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "figures", f"predictions_{pattern_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_bars(all_results):
    """Bar chart comparing RMSE, MAE, and Score across all patterns and models."""
    rows = []
    for pattern, results in all_results.items():
        for model, r in results.items():
            rows.append({
                "Pattern": pattern,
                "Model": model,
                "RMSE": r["RMSE"],
                "MAE": r["MAE"],
                "Score": r["Score"],
            })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric in zip(axes, ["RMSE", "MAE", "Score"]):
        pivot = df.pivot(index="Pattern", columns="Model", values=metric)
        pivot.plot(kind="bar", ax=ax, color=[COLORS.get(m, "#999") for m in pivot.columns])
        ax.set_title(metric, fontweight="bold", fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel(metric + (" (%)" if metric == "Score" else ""))
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Model Comparison Across Workload Patterns",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "figures", "summary_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    include_novelty = "--original-only" not in sys.argv
    mode_label = "Paper + Novelty (5 models)" if include_novelty else "Paper Only (3 models)"

    print("=" * 70)
    print(f"  MTTD Framework Benchmark - {mode_label}")
    print(f"  Duration: {DURATION_MIN} min | Train: {TRAIN_FRAC*100:.0f}% | "
          f"Test: {TEST_LEN} pts | Window: {WINDOW}")
    print("=" * 70)

    ensure_results_dir()

    # 1. Plot workload patterns (Fig. 1)
    print("\n[1/4] Generating workload pattern plots (Fig. 1)...")
    plot_workload_patterns()

    # 2. Run all models on all patterns
    all_results = {}
    all_tables = {}

    print("\n[2/4] Running models on all 4 workload patterns...")
    for pattern_name, generator in PATTERNS.items():
        print(f"\n{'-' * 50}")
        print(f"  Pattern: {pattern_name}")
        print(f"{'-' * 50}")

        df = generator(duration_min=DURATION_MIN)
        results = evaluate_all_models(
            df, train_frac=TRAIN_FRAC, test_len=TEST_LEN, window=WINDOW,
            include_novelty=include_novelty,
        )
        all_results[pattern_name] = results

        # Print table
        table = results_to_table(results)
        all_tables[pattern_name] = table
        print(table.to_string())

        # Select best model
        active, scores = select_active_model(results)
        print(f"\n  >>> Active model: {active} (Score={scores[active]:.2f}%)")

        # Save CSV
        csv_path = os.path.join(RESULTS_DIR, "tables", f"table_{pattern_name}.csv")
        table.to_csv(csv_path)

        # Plot predictions for this pattern immediately
        plot_predictions(pattern_name, results)

    # 3. Summary comparison
    print("\n[3/4] Generating summary comparison bar chart...")
    plot_summary_bars(all_results)

    # Build grand summary table
    print("\n[4/4] Generating grand summary table...")
    summary_rows = []
    for pattern, results in all_results.items():
        active, scores = select_active_model(results)
        for model, r in results.items():
            summary_rows.append({
                "Pattern": pattern,
                "Model": model,
                "RMSE": round(r["RMSE"], 4),
                "MAE": round(r["MAE"], 4),
                "MAPE": f"{r['MAPE']:.2f}%" if "MAPE" in r and not np.isnan(r["MAPE"]) else "N/A",
                "Score(%)": round(r["Score"], 2),
                "Best?": "[BEST]" if model == active else "",
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "tables", "grand_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 70}")
    print("  GRAND SUMMARY")
    print(f"{'=' * 70}")
    print(summary_df.to_string(index=False))
    print(f"\nAll results saved to: {RESULTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
