"""
verify_dataset.py - Validation script for the canonical cricket hourly workload dataset.

Checks:
  1. Row count is exactly 1,344 (56 days x 24 hours).
  2. Timestamps are monotonic hourly increments starting 2023-10-01 00:00:00.
  3. Hits are non-negative integers with no NaN or missing values.
  4. Daily sums match data/raw/daily_totals_recovered.csv exactly.
  5. SHA-256 hash matches 0411f7b72cab109e31d0eed31da0c385c831ff85796dea49905b8b0926140c45.
  6. Computes and prints the oracle noise-floor Score (4.06%).
"""
import csv
import hashlib
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Ensure repository root is on sys.path to import arima_baseline
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arima_baseline import performance_score

EXPECTED_SHA256 = "0411f7b72cab109e31d0eed31da0c385c831ff85796dea49905b8b0926140c45"
EXPECTED_ROWS = 1344
START_TIMESTAMP = datetime(2023, 10, 1, 0, 0, 0)
DATASET_CSV = SCRIPT_DIR / "cricket_hourly_workload.csv"
RECOVERED_CSV = SCRIPT_DIR / "raw" / "daily_totals_recovered.csv"

PEAK_START, PEAK_END = 14, 22
PEAK_SHARE = 0.70


def verify():
    print(f"Verifying dataset: {DATASET_CSV}")
    assert DATASET_CSV.exists(), f"Dataset file missing: {DATASET_CSV}"
    assert RECOVERED_CSV.exists(), f"Recovered daily totals missing: {RECOVERED_CSV}"

    # 1. SHA-256
    with open(DATASET_CSV, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert file_sha256 == EXPECTED_SHA256, (
        f"SHA-256 mismatch!\n  Expected: {EXPECTED_SHA256}\n  Got:      {file_sha256}"
    )
    print(f"  [PASS] SHA-256 hash: {file_sha256}")

    # 2. Read dataset and check schema, monotonic timestamps, non-negative values
    timestamps = []
    hits_list = []
    daily_sums = defaultdict(int)

    with open(DATASET_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["Timestamp", "Hits"], f"Unexpected header: {header}"

        prev_dt = None
        for i, row in enumerate(reader):
            assert len(row) == 2, f"Row {i+1} invalid column count: {row}"
            ts_str, hits_str = row[0].strip(), row[1].strip()

            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            if i == 0:
                assert dt == START_TIMESTAMP, f"First timestamp is {dt}, expected {START_TIMESTAMP}"
            else:
                expected_dt = prev_dt + timedelta(hours=1)
                assert dt == expected_dt, f"Timestamp discontinuity at row {i+1}: got {dt}, expected {expected_dt}"
            prev_dt = dt

            hits_float = float(hits_str)
            assert not np.isnan(hits_float), f"NaN hit value at row {i+1}: {ts_str}"
            assert hits_float >= 0, f"Negative hit value at row {i+1}: {hits_str}"
            assert hits_float.is_integer(), f"Non-integer hit value at row {i+1}: {hits_str}"

            hits = int(hits_float)
            timestamps.append(dt)
            hits_list.append(hits)
            daily_sums[dt.strftime("%Y-%m-%d")] += hits

    row_count = len(hits_list)
    assert row_count == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {row_count}"
    print(f"  [PASS] Row count: {row_count} (56 days x 24 hours)")
    print("  [PASS] Timestamps are strictly monotonic hourly increments without gaps")
    print(f"  [PASS] Hits range: [{min(hits_list):,}, {max(hits_list):,}] with no NaN or negative values")

    # 3. Daily sums match daily_totals_recovered.csv exactly
    recovered_totals = {}
    with open(RECOVERED_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row or not row[0].strip():
                continue
            recovered_totals[row[0].strip()] = int(row[1].strip())

    assert len(daily_sums) == len(recovered_totals), (
        f"Day count mismatch: {len(daily_sums)} in workload vs {len(recovered_totals)} in recovered CSV"
    )

    for d, total in recovered_totals.items():
        assert d in daily_sums, f"Date {d} missing from hourly workload"
        assert daily_sums[d] == total, f"Date {d} sum mismatch: hourly {daily_sums[d]} != recovered {total}"
    print(f"  [PASS] Daily totals preserved exactly across all {len(recovered_totals)} days")

    # 4. Oracle noise-floor score
    n_peak = PEAK_END - PEAK_START
    weights = np.full(24, (1 - PEAK_SHARE) / (24 - n_peak))
    weights[PEAK_START:PEAK_END] = PEAK_SHARE / n_peak

    actual_hits = np.array(hits_list, dtype=float)
    oracle_preds = np.zeros(row_count, dtype=float)
    for idx, dt in enumerate(timestamps):
        d_str = dt.strftime("%Y-%m-%d")
        h = dt.hour
        oracle_preds[idx] = recovered_totals[d_str] * weights[h]

    oracle_score = performance_score(actual_hits, oracle_preds)
    print(f"\nOracle Noise-Floor Score: {oracle_score:.2f}% (exact: {oracle_score:.4f}%)")
    assert abs(oracle_score - 4.06) < 0.01, f"Oracle score {oracle_score} != 4.06%"
    print("All assertions passed successfully.")
    return True


if __name__ == "__main__":
    verify()
