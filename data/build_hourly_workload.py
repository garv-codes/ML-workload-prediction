#!/usr/bin/env python3
"""
build_hourly_workload.py — Convert daily Wikimedia page-view counts into an
hourly workload trace suitable for the MTTD framework.

Data source
-----------
Daily page-view counts for the **2023 Cricket World Cup** article on
English Wikipedia, retrieved from the Wikimedia Pageviews REST API.

    API endpoint : https://wikimedia.org/api/rest_v1/metrics/pageviews/
                   per-article/{project}/{access}/{agent}/{article}/
                   daily/{start}/{end}

    Default parameters used (override with CLI flags):
        project  : en.wikipedia
        article  : 2023_Cricket_World_Cup   TODO(owner): confirm exact title
        access   : all-access               TODO(owner): confirm filter used
        agent    : all-agents               TODO(owner): confirm filter used
        start    : 20231005
        end      : 20231130                 (57 daily points, Oct 5 – Nov 30)

Transformation
--------------
Each of the first 56 daily totals is expanded into 24 hourly buckets:

    * Peak hours  14:00–21:00 (8 hours, end exclusive at 22:00): receive 70 %
      of the day's views, distributed proportionally via a raised-cosine
      profile centred on 18:00 IST.
    * Off-peak hours (remaining 16 hours): receive 30 %, uniformly.

Multiplicative log-normal noise is applied per hour to simulate flash-crowd
spikes, then each day's hours are **renormalised** so the daily sum is
preserved exactly (to within 1 view due to rounding).

Timezone assumption: the 14:00-22:00 window is in IST (UTC+05:30), matching
the host country for the 2023 Cricket World Cup (India).  Wikimedia daily
boundaries are UTC; the script does NOT shift day boundaries.  This means
the "match hours" window is approximate — the synthetic intra-day shape is
not a measured quantity.  TODO(owner): confirm timezone used in original run.

The 57th daily data point (Nov 30) provides a trailing day but is NOT
expanded into hourly rows, because 56 days × 24 h = 1,344 rows, matching
the committed dataset row count.

Determinism
-----------
All randomness uses numpy.random.default_rng(seed).  The default seed is 42.
Running with the same seed always produces a byte-identical output file.

Usage
-----
    # Offline (uses cached raw CSV):
    python data/build_hourly_workload.py --seed 42

    # Refresh raw data from Wikimedia API:
    python data/build_hourly_workload.py --refresh --seed 42

    # Custom article / date range:
    python data/build_hourly_workload.py --article "2023_Cricket_World_Cup" \\
        --start 20231005 --end 20231130 --seed 42
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_CSV = SCRIPT_DIR / "raw" / "wikimedia_daily.csv"
OUT_CSV = SCRIPT_DIR / "cricket_hourly_workload.csv"

# ---------------------------------------------------------------------------
# Wikimedia API defaults
# ---------------------------------------------------------------------------
DEFAULT_PROJECT = "en.wikipedia"
DEFAULT_ARTICLE = "2023_Cricket_World_Cup"  # TODO(owner): confirm exact article title
DEFAULT_ACCESS = "all-access"               # TODO(owner): confirm access filter
DEFAULT_AGENT = "all-agents"                # TODO(owner): confirm agent filter
DEFAULT_START = "20231005"
DEFAULT_END = "20231130"
USER_AGENT = (
    "MLWorkloadPrediction/1.0 "
    "(https://github.com/garv-codes/ML-workload-prediction; academic research)"
)

# ---------------------------------------------------------------------------
# Hourly profile constants
# ---------------------------------------------------------------------------
PEAK_START = 14   # hour (inclusive), IST
PEAK_END = 22     # hour (exclusive), IST — so peak hours are 14,15,...,21
PEAK_FRAC = 0.70  # 70 % of daily views in peak hours
OFFPEAK_FRAC = 0.30
DAYS_TO_EXPAND = 56  # 56 days × 24 h = 1,344 hourly rows
EXPECTED_ROWS = DAYS_TO_EXPAND * 24  # 1,344
NOISE_SIGMA = 0.15   # std-dev for log-normal multiplicative noise


def fetch_from_api(project, article, access, agent, start, end):
    """Fetch daily page-view counts from the Wikimedia Pageviews REST API."""
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{project}/{access}/{agent}/{article}/daily/{start}/{end}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"Fetching from: {url}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    rows = []
    for item in data["items"]:
        date_str = item["timestamp"][:8]
        views = int(item["views"])
        rows.append((date_str, views))
    return rows


def load_raw_csv(path):
    """Load cached daily data from CSV."""
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["date"].strip(), int(row["views"].strip())))
    return rows


def save_raw_csv(rows, path):
    """Cache daily data to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "views"])
        for date_str, views in rows:
            writer.writerow([date_str, views])
    print(f"Saved raw daily CSV: {path}  ({len(rows)} rows)")


def build_hourly(daily_rows, seed=42):
    """Expand daily totals into hourly rows with a synthetic profile.

    For each of the first DAYS_TO_EXPAND days:
      1. Compute a base hourly profile:
         - Peak hours (14:00-21:00, 8 hours): 70 % of daily views,
           weighted by a raised-cosine centred at 18:00.
         - Off-peak hours (remaining 16 hours): 30 % of daily views,
           uniformly distributed.
      2. Apply multiplicative log-normal noise to each hour.
      3. Renormalise so hourly values sum exactly to the daily total.
      4. Round to non-negative integers; adjust the largest hour by
         the rounding residual so the daily sum is preserved exactly.

    Returns a list of (timestamp_str, hits_int) tuples.
    """
    rng = np.random.default_rng(seed)

    # Pre-compute the base profile weights for 24 hours
    peak_hours = list(range(PEAK_START, PEAK_END))  # [14..21]
    n_peak = len(peak_hours)   # 8
    n_offpeak = 24 - n_peak    # 16

    # Raised-cosine weights for peak hours (centred on hour 18)
    peak_centre = (PEAK_START + PEAK_END - 1) / 2.0  # 17.5
    cosine_weights = np.array([
        1.0 + np.cos(np.pi * (h - peak_centre) / (n_peak / 2.0))
        for h in peak_hours
    ])
    cosine_weights /= cosine_weights.sum()  # normalise to sum=1 within peak

    hourly_records = []
    for day_idx in range(DAYS_TO_EXPAND):
        date_str, daily_views = daily_rows[day_idx]
        day_date = datetime.strptime(date_str, "%Y%m%d")

        # Base profile (24 floats summing to daily_views)
        base = np.zeros(24)
        for i, h in enumerate(peak_hours):
            base[h] = PEAK_FRAC * daily_views * cosine_weights[i]
        offpeak_share = OFFPEAK_FRAC * daily_views / n_offpeak
        for h in range(24):
            if h not in peak_hours:
                base[h] = offpeak_share

        # Multiplicative log-normal noise
        noise = rng.lognormal(mean=0.0, sigma=NOISE_SIGMA, size=24)
        noisy = base * noise

        # Renormalise to preserve daily total exactly
        noisy_sum = noisy.sum()
        if noisy_sum > 0:
            noisy = noisy * (daily_views / noisy_sum)
        else:
            # Degenerate case: spread uniformly
            noisy = np.full(24, daily_views / 24.0)

        # Round to non-negative integers
        hourly_int = np.clip(np.round(noisy), 0, None).astype(int)

        # Fix rounding residual: adjust the largest hour
        residual = daily_views - hourly_int.sum()
        if residual != 0:
            idx_max = int(np.argmax(hourly_int))
            hourly_int[idx_max] += residual

        # Assert daily sum preserved
        assert abs(hourly_int.sum() - daily_views) <= 1, (
            f"Day {date_str}: hourly sum {hourly_int.sum()} != daily {daily_views}"
        )

        # Emit 24 rows
        for h in range(24):
            ts = day_date + timedelta(hours=h)
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            hourly_records.append((ts_str, int(hourly_int[h])))

    return hourly_records


def write_hourly_csv(records, path):
    """Write the hourly workload CSV (Timestamp,Hits)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Hits"])
        for ts, hits in records:
            writer.writerow([ts, hits])
    print(f"Saved hourly CSV: {path}  ({len(records)} rows)")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Build the semi-synthetic hourly workload CSV from "
                    "Wikimedia daily page-view data."
    )
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch daily data from the Wikimedia API "
                             "(default: use cached raw CSV)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for deterministic noise (default: 42)")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help=f"Wikimedia project (default: {DEFAULT_PROJECT})")
    parser.add_argument("--article", default=DEFAULT_ARTICLE,
                        help=f"Article title (default: {DEFAULT_ARTICLE})")
    parser.add_argument("--access", default=DEFAULT_ACCESS,
                        help=f"Access filter (default: {DEFAULT_ACCESS})")
    parser.add_argument("--agent", default=DEFAULT_AGENT,
                        help=f"Agent filter (default: {DEFAULT_AGENT})")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"Start date YYYYMMDD (default: {DEFAULT_START})")
    parser.add_argument("--end", default=DEFAULT_END,
                        help=f"End date YYYYMMDD (default: {DEFAULT_END})")
    parser.add_argument("--out", type=str, default=None,
                        help=f"Output CSV path (default: {OUT_CSV})")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else OUT_CSV

    # Step 1: obtain daily data
    if args.refresh or not RAW_CSV.exists():
        print("Fetching daily data from Wikimedia Pageviews API...")
        daily_rows = fetch_from_api(
            args.project, args.article, args.access, args.agent,
            args.start, args.end
        )
        save_raw_csv(daily_rows, RAW_CSV)
    else:
        print(f"Using cached raw CSV: {RAW_CSV}")
        daily_rows = load_raw_csv(RAW_CSV)

    print(f"Loaded {len(daily_rows)} daily data points")

    if len(daily_rows) < DAYS_TO_EXPAND:
        print(f"ERROR: need at least {DAYS_TO_EXPAND} daily points, "
              f"got {len(daily_rows)}", file=sys.stderr)
        sys.exit(1)

    # Step 2: expand to hourly
    print(f"\nExpanding first {DAYS_TO_EXPAND} days into hourly records "
          f"(seed={args.seed})...")
    hourly_records = build_hourly(daily_rows, seed=args.seed)

    assert len(hourly_records) == EXPECTED_ROWS, (
        f"Expected {EXPECTED_ROWS} rows, got {len(hourly_records)}"
    )

    # Validate: no NaN, non-negative, integer
    for ts, hits in hourly_records:
        assert isinstance(hits, int), f"Non-integer hits: {hits} at {ts}"
        assert hits >= 0, f"Negative hits: {hits} at {ts}"

    # Step 3: write output
    write_hourly_csv(hourly_records, out_path)

    # Step 4: report hash for reproducibility
    digest = sha256_file(out_path)
    print(f"\nSHA-256: {digest}")
    print(f"Rows:    {len(hourly_records)}")
    print("Done.")


if __name__ == "__main__":
    main()
