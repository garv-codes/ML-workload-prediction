# Dataset: Cricket Hourly Workload

## Source

Daily page-view counts for the English Wikipedia article on the
[2023 Cricket World Cup](https://en.wikipedia.org/wiki/2023_Cricket_World_Cup),
retrieved from the **Wikimedia Pageviews REST API**.

### API endpoint and parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Endpoint  | `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/` | |
| Project   | `en.wikipedia` | English Wikipedia |
| Article   | `2023_Cricket_World_Cup` | TODO(owner): confirm exact title used |
| Access    | `all-access` | TODO(owner): confirm filter used |
| Agent     | `all-agents` | TODO(owner): confirm filter used |
| Granularity | `daily` | |
| Start     | `20231005` | First day of the World Cup |
| End       | `20231130` | 57 daily data points (Oct 5 – Nov 30, 2023) |

### Retrieval date

The cached raw data in `raw/wikimedia_daily.csv` was retrieved in October 2026.
TODO(owner): confirm or update the original retrieval date.

### Licensing and attribution

Wikimedia page-view data is provided by the Wikimedia Foundation under the
[Wikimedia REST API Terms of Use](https://wikimedia.org/api/rest_v1/#/Pageviews_data).
Page-view statistics are aggregated, non-personal data.  Please see
[Wikimedia's Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)
and the [data dumps license page](https://dumps.wikimedia.org/legal.html) for
authoritative licensing information.  We do not assert a specific license for
the raw page-view counts; consult Wikimedia's own terms.

---

## What is real / what is synthetic

| Aspect | Status | Details |
|--------|--------|---------|
| **Daily page-view totals** | **Real** (measured) | 57 daily counts from Wikimedia, reflecting actual user visits to the article. |
| **Hourly split (70/30 peak/off-peak)** | **Synthetic** (modelled) | 70 % of each day's views assigned to hours 14:00–21:00 (IST), 30 % to remaining hours. |
| **Within-day shape** | **Synthetic** | Raised-cosine profile centred on 18:00 for peak hours; uniform for off-peak. |
| **Flash-crowd noise and spikes** | **Synthetic** | Multiplicative log-normal noise (σ = 0.15) injected per hour. |
| **Daily total preservation** | **Exact** | After noise, hourly values are renormalised so each day's sum equals the real daily total (within ±1 due to integer rounding). |
| **Seed** | **Fixed (42)** | Deterministic; same seed → identical output. |

> **Important:** Only the daily totals are real measurements.  The intra-day
> (hourly) shape and the spikes are synthetic, produced by the build script.
> This dataset should be described as **"semi-synthetic"** or **"real daily
> totals with a synthetic hourly profile"** — never as "real hourly traffic"
> or "real-world traffic dynamics".

---

## Transformation steps

1. **Fetch daily data** from the Wikimedia Pageviews API (or read cached
   `raw/wikimedia_daily.csv`).  57 daily points: 2023-10-05 to 2023-11-30.

2. **Expand the first 56 days** into 24 hourly buckets each (56 × 24 = 1,344 rows).
   The 57th day (Nov 30) is not expanded because the target row count is 1,344.

3. **Compute base hourly profile** for each day:
   - Peak hours 14:00–21:00 (8 hours, end exclusive at 22:00): receive 70 %
     of the daily total, distributed via a raised-cosine profile centred at 18:00.
   - Off-peak hours (remaining 16): receive 30 %, uniformly.

4. **Apply multiplicative log-normal noise** (μ=0, σ=0.15) to each hour.

5. **Renormalise** hourly values so they sum exactly to the day's real total.

6. **Round** to non-negative integers; adjust the largest hour by the rounding
   residual to preserve the daily sum exactly.

7. **Assert** each day's hourly sum equals the raw daily value within ±1.

### Timezone assumption

The peak window 14:00–22:00 is intended to model IST (UTC+05:30), since
matches were played in India.  Wikimedia daily boundaries are UTC.  The
script does **not** shift day boundaries to align with IST — the peak window
is approximate and synthetic regardless.
TODO(owner): confirm the timezone assumption used in the original run.

---

## Output schema

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | `str` (`YYYY-MM-DD HH:MM:SS`) | Hourly timestamp, starting 2023-10-05 00:00:00 |
| `Hits` | `int` (non-negative) | Page views attributed to this hour |

- **Rows:** 1,344 (56 days × 24 hours)
- **No NaN, no trailing junk lines, no negative values.**
- Header row: `Timestamp,Hits`

---

## File inventory

| File | Description |
|------|-------------|
| `raw/wikimedia_daily.csv` | Cached daily page-view data (57 rows, `date,views`) |
| `build_hourly_workload.py` | Generator script — run with `--seed 42` for the canonical CSV |
| `cricket_hourly_workload.csv` | **Canonical** semi-synthetic hourly workload (1,344 rows) |

---

## Regeneration

```bash
# Offline (uses cached raw CSV):
python data/build_hourly_workload.py --seed 42

# Refresh from API:
python data/build_hourly_workload.py --refresh --seed 42
```

Running with `--seed 42` always produces an identical file (same SHA-256 hash).
