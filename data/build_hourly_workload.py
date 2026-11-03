"""
build_hourly_workload.py - convert daily Wikipedia page views into an hourly
workload trace for the MTTD / CloudSim experiments.

Original generator script for cricket_hourly_workload.csv.

Input : pageviews CSV exported from https://pageviews.wmcloud.org
        (columns: Date, <article name>), one row per day.
Output: cricket_hourly_workload.csv with columns Timestamp, Hits
        (24 rows per input day).

Method (per day):
  1. Base profile: 70% of the day's views spread evenly over the match window
     14:00-21:59 (8 hours, 8.75% each), 30% over the other 16 hours (1.875% each).
  2. Multiply every hour by uniform random noise in [0.92, 1.08] (+/-8%).
  3. Renormalise so the 24 hours add up to the real daily total again.
  4. Round to whole numbers and give the rounding remainder to the largest
     hour, so each day's sum equals the real daily count exactly.

A fixed seed makes the output identical on every run.

Usage:
    python data/build_hourly_workload.py            # reads data/raw/pageviews-2023-10-01-2023-11-25.csv
"""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PEAK_START, PEAK_END = 14, 22        # match window 14:00-21:59 (end exclusive)
PEAK_SHARE = 0.70                    # 70% of daily views in the match window
NOISE_LOW, NOISE_HIGH = 0.92, 1.08   # +/-8% uniform noise
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "raw" / "pageviews-2023-10-01-2023-11-25.csv"
DEFAULT_OUT = SCRIPT_DIR / "cricket_hourly_workload.csv"


def read_daily(path):
    """Read (date, views) pairs straight from the pageviews export."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)                                 # header: Date,<article>
        for line in reader:
            if len(line) < 2 or not line[1].strip():
                continue
            rows.append((datetime.strptime(line[0].strip(), "%Y-%m-%d"),
                         int(line[1].strip())))
    return rows


def hourly_profile():
    """24 weights summing to 1: 8.75% per match hour, 1.875% per other hour."""
    n_peak = PEAK_END - PEAK_START
    w = np.full(24, (1 - PEAK_SHARE) / (24 - n_peak))
    w[PEAK_START:PEAK_END] = PEAK_SHARE / n_peak
    return w


def build(daily, seed=SEED):
    rng = np.random.default_rng(seed)
    weights = hourly_profile()
    out = []
    for day, total in daily:
        noisy = total * weights * rng.uniform(NOISE_LOW, NOISE_HIGH, 24)
        noisy *= total / noisy.sum()                 # keep the real daily total
        hours = np.round(noisy).astype(int)
        hours[np.argmax(hours)] += total - hours.sum()   # fix rounding remainder
        assert hours.sum() == total and (hours >= 0).all()
        for h in range(24):
            out.append(((day + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S"),
                        int(hours[h])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help=f"path to daily pageviews CSV (default: {DEFAULT_INPUT})")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output hourly CSV path (default: {DEFAULT_OUT})")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    daily = read_daily(a.input)
    hourly = build(daily, a.seed)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Timestamp", "Hits"])
        w.writerows(hourly)

    hits = [h for _, h in hourly]
    peak = int(np.argmax(hits))
    print(f"Days read      : {len(daily)} ({daily[0][0]:%d %b %Y} - {daily[-1][0]:%d %b %Y})")
    print(f"Hourly rows    : {len(hourly)}")
    print(f"Total views    : {sum(hits):,} (daily file total {sum(v for _, v in daily):,})")
    print(f"Peak hour      : {hits[peak]:,} at {hourly[peak][0]} (hour index {peak})")
    print(f"Saved          : {a.out}")


if __name__ == "__main__":
    main()
