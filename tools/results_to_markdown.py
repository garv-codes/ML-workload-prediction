"""
tools/results_to_markdown.py - Generate markdown tables from result CSVs.

Reads the benchmark tables, MTTD bridge results, and model summaries,
and prints markdown tables suitable for inclusion in the README.

Usage:
    python tools/results_to_markdown.py
"""
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
TABLES = RESULTS / "tables"


def csv_to_markdown(rows, headers=None):
    """Convert a list of dicts (or list of lists with headers) to a markdown table."""
    if not rows:
        return "(no data)\n"
    if headers is None:
        headers = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        if isinstance(row, dict):
            vals = [str(row.get(h, "")) for h in headers]
        else:
            vals = [str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def read_csv_file(path):
    """Read a CSV and return list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def section_benchmark_tables():
    """Section 9.1: Synthetic benchmark tables per pattern."""
    print("### 9.1 Synthetic Workload Benchmarks\n")
    patterns = ["uniform", "linear_low", "linear_high", "random"]
    for pat in patterns:
        csv_path = TABLES / f"table_{pat}.csv"
        if not csv_path.exists():
            print(f"**{pat}**: (not found)\n")
            continue
        rows = read_csv_file(csv_path)
        print(f"**{pat.replace('_', ' ').title()}**\n")
        print(csv_to_markdown(rows))
        print()

    # Grand summary
    grand = TABLES / "grand_summary.csv"
    if grand.exists():
        print("**Grand Summary (all patterns)**\n")
        rows = read_csv_file(grand)
        print(csv_to_markdown(rows))
        print()


def section_mttd_results():
    """Section 9.2: MTTD bridge results."""
    print("### 9.2 MTTD Bridge Results (Cricket Hourly Workload)\n")

    # Model summary
    summary_path = RESULTS / "predictions_retrain_model_summary.csv"
    if summary_path.exists():
        print("**Per-Model Summary (with retraining)**\n")
        rows = read_csv_file(summary_path)
        print(csv_to_markdown(rows))
        print()

    # Retrain events
    scores_path = RESULTS / "predictions_retrain_model_scores.csv"
    if scores_path.exists():
        rows = read_csv_file(scores_path)
        retrain_events = [r for r in rows if r.get("retrain", "0") == "1"]
        if retrain_events:
            print("**Retrain Events**\n")
            headers = ["phase", "start_hour", "model", "Score", "threshold"]
            print(csv_to_markdown(retrain_events, headers))
            print()
        else:
            print("**Retrain Events**: None\n")

    # MTTD vs no-retrain comparison
    retrain_path = RESULTS / "predictions_retrain.csv"
    noretrain_path = RESULTS / "predictions_noretrain.csv"

    if retrain_path.exists() and noretrain_path.exists():
        print("**MTTD Retrain vs No-Retrain Comparison**\n")
        retrain_rows = read_csv_file(retrain_path)
        noretrain_rows = read_csv_file(noretrain_path)

        def compute_overall_metrics(rows):
            import math
            actual = [float(r["actual"]) for r in rows]
            pred = [float(r["predicted"]) for r in rows]
            n = len(actual)
            if n == 0:
                return {}
            mse = sum((a - p) ** 2 for a, p in zip(actual, pred)) / n
            r = math.sqrt(mse)
            m = sum(abs(a - p) for a, p in zip(actual, pred)) / n
            mean_y = sum(actual) / n
            rms_y = math.sqrt(sum(a ** 2 for a in actual) / n)
            if rms_y > 0 and mean_y > 0:
                score = 100.0 * (0.5 * r / rms_y + 0.5 * m / mean_y)
            else:
                score = float("inf")
            return {"RMSE": f"{r:.2f}", "MAE": f"{m:.2f}", "Score(%)": f"{score:.2f}"}

        retrain_metrics = compute_overall_metrics(retrain_rows)
        noretrain_metrics = compute_overall_metrics(noretrain_rows)

        comp_rows = [
            {"Mode": "With Retraining", **retrain_metrics},
            {"Mode": "Without Retraining", **noretrain_metrics},
        ]
        print(csv_to_markdown(comp_rows, ["Mode", "RMSE", "MAE", "Score(%)"]))
        print()

    # Oracle noise floor
    print("**Oracle Noise Floor**: 4.06%\n")
    print("The noise floor represents the theoretical minimum prediction error\n"
          "achievable on this dataset, limited by the synthetic uniform ±8% noise\n"
          "injected during hourly profile generation.\n")


def section_mttd_vs_best_fixed():
    """Compare MTTD active model vs best fixed single model."""
    summary_path = RESULTS / "predictions_retrain_model_summary.csv"
    if not summary_path.exists():
        return

    rows = read_csv_file(summary_path)
    if not rows:
        return

    print("**MTTD Active Model vs Best Fixed Model**\n")

    # Find best fixed model (lowest overall_Score)
    best_fixed = min(rows, key=lambda r: float(r.get("overall_Score", "inf")))
    # MTTD score = weighted average by hours_active
    total_hours = sum(int(r.get("hours_active", 0)) for r in rows)
    if total_hours > 0:
        mttd_score = sum(
            float(r.get("overall_Score", 0)) * int(r.get("hours_active", 0))
            for r in rows
        ) / total_hours
    else:
        mttd_score = float("inf")

    comp_rows = [
        {"Strategy": "MTTD (dynamic)", "Overall Score(%)": f"{mttd_score:.2f}",
         "Notes": "Weighted by hours active"},
        {"Strategy": f"Best Fixed ({best_fixed.get('model', '?')})",
         "Overall Score(%)": best_fixed.get("overall_Score", "?"),
         "Notes": f"{best_fixed.get('cycles_best', '?')} cycles best"},
    ]
    print(csv_to_markdown(comp_rows, ["Strategy", "Overall Score(%)", "Notes"]))
    print()


def main():
    print("# Generated Result Tables\n")
    print("_Auto-generated by `tools/results_to_markdown.py`_\n")
    print("---\n")

    section_benchmark_tables()

    print("---\n")

    section_mttd_results()
    section_mttd_vs_best_fixed()


if __name__ == "__main__":
    main()
