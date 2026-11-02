# Cricket Hourly Workload Dataset

This directory contains the **canonical semi-synthetic cricket workload dataset**
used for evaluating the MTTD framework on multi-day real-world traffic volume.

---

## 1. Provenance and Metadata

| Attribute | Value |
|---|---|
| **Source** | Daily page views exported from the Wikimedia Pageviews Analysis tool (`pageviews.wmcloud.org`), columns `Date,<article name>` |
| **Article title** | `TODO(owner)`: confirm exact article title used in initial query |
| **Project** | `TODO(owner)`: confirm project filter (e.g. `en.wikipedia`) |
| **Access filter** | `TODO(owner)`: confirm access filter (e.g. `all-access`) |
| **Agent filter** | `TODO(owner)`: confirm agent filter (e.g. `user`) |
| **Export date** | `TODO(owner)`: confirm date export was downloaded |
| **Time period** | **2023-10-01 to 2023-11-25** (56 days) |
| **Granularity** | Hourly (56 × 24 = 1,344 rows) |
| **File schema** | `Timestamp,Hits` (e.g. `2023-10-01 00:00:00,759`) |
| **First timestamp** | `2023-10-01 00:00:00` |
| **Last timestamp** | `2023-11-25 23:00:00` |
| **SHA-256** | `0411f7b72cab109e31d0eed31da0c385c831ff85796dea49905b8b0926140c45` |
| **Total hits** | 29,957,762 |
| **Mean hits/hour** | 22,290.0 |
| **Min hits/hour** | 630 |
| **Max hits/hour** | 117,612 (2023-11-19 17:00:00) |
| **Day 1 total** | 111,272 |
| **Day 56 total** | 36,908 |
| **Oracle noise floor** | **4.06%** (Eq. 4 performance score) |

---

## 2. Real vs. Synthetic Components

| Component | Status | Description |
|---|---|---|
| **Daily page-view totals** | **Real** (measured) | The 56 daily totals are empirical page-view counts from Wikimedia. They are preserved **exactly** in the hourly file (daily sum of hourly hits = daily total). |
| **Intra-day split** | **Synthetic** (modelled) | 70% of each day's views are allocated evenly over the 8-hour window 14:00–21:59 (8.75% per hour). The remaining 30% are allocated evenly over the other 16 hours (1.875% per hour). |
| **Within-band shape** | **Synthetic** (flat) | Flat distribution within peak and off-peak bands. No curved or bell-shaped shaping is applied. |
| **Noise injection** | **Synthetic** | Each hour is multiplied by uniform noise in `[0.92, 1.08]` (uniform ±8%), seeded with RNG seed 42. |
| **Renormalisation** | **Exact** | Hourly values are scaled to match the empirical daily total exactly, rounded to integers, and any rounding residual is assigned to the largest hour. |

> **Critical Notice:** Only daily totals represent real empirical measurements.
> The hourly profile and fluctuations are synthetically generated.
> This trace is **semi-synthetic** and should not be cited as empirical hourly server traffic.

---

## 3. Timezone Note

The 14:00–21:59 peak window was chosen to approximate local match hours in India (IST, UTC+05:30)
during the cricket tournament. Wikimedia Pageviews daily boundaries are recorded in UTC.
The generation script does not shift calendar day boundaries to align with IST; the intra-day
profile is synthetic in either case.

---

## 4. Recovered Daily Totals

Because the build script guarantees that the sum of hourly values for each day
equals the daily total, summing the canonical hourly CSV reproduces the original
daily totals with zero discrepancy.

The recovered totals are saved in [`raw/daily_totals_recovered.csv`](raw/daily_totals_recovered.csv)
(56 rows, `Date,views`).

---

## 5. Dataset History

1. **Commit `808162d`:** Earlier `cricket_hourly_workload.csv` with integer index `Timestamp`
   (1..1344) and a trailing termination banner (`** Process exited **`). Its daily totals
   diverged from the canonical data (e.g. Day 1 was 110,592 instead of 111,272; differences up to 2.7×).
   All previously committed `results/predictions_*` were generated against that obsolete version.
2. **Commit `0a2a93c`:** An automated rebuild that overwrote the data with an incompatible date range
   and non-canonical parameters. That rebuild was discarded.
3. **Canonical Version:** The current `data/cricket_hourly_workload.csv` (1,344 rows,
   SHA-256 `0411f7b72cab109e31d0eed31da0c385c831ff85796dea49905b8b0926140c45`).
   All experiments and results in this repository are executed on this version.

---

## 6. Verification and Reproduction

### Verification

Run the verification test suite:

```bash
python data/verify_dataset.py
```

The script verifies:
1. Row count is exactly 1,344.
2. Timestamps are monotonically increasing hourly timestamps from `2023-10-01 00:00:00` to `2023-11-25 23:00:00`.
3. Non-negative integers with no missing or NaN values.
4. Daily totals match `raw/daily_totals_recovered.csv` with 0 difference.
5. SHA-256 hash equals `0411f7b72cab109e31d0eed31da0c385c831ff85796dea49905b8b0926140c45`.
6. Prints the oracle noise floor Score (4.06%).

### Regeneration from Daily Totals

```bash
python data/build_hourly_workload.py --input data/raw/daily_totals_recovered.csv --out data/cricket_hourly_workload.csv --seed 42
```

Re-running with seed 42 reproduces every value in `cricket_hourly_workload.csv` byte-identically.

---

## 7. Irreducible Noise Floor

Because uniform noise in `[0.92, 1.08]` (±8%) is injected into every hour,
an oracle predictor that knows the exact daily total and the exact deterministic profile
still incurs prediction error against the noisy realization:

$$\text{Oracle Noise Floor Score} \approx 4.06\%$$

Scores near 4% represent this irreducible limit; differences between models near 4%
reflect noise fitting rather than meaningful architectural superiority.

---

## 8. Terms and Attribution

Wikimedia page view counts are made available under Wikimedia Terms of Use:
- [Wikimedia Terms of Use](https://wikimediafoundation.org/wiki/Terms_of_Use)
- [Wikimedia Pageviews API Documentation](https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews)
