# Cricket Hourly Workload Dataset

This directory contains the **canonical semi-synthetic cricket workload dataset**
used for evaluating the MTTD framework on multi-day real-world traffic volume.

---

## 1. Source and Export Settings

| Attribute | Value |
|---|---|
| **Tool** | [Wikimedia Pageviews Analysis](https://pageviews.wmcloud.org/) (manual web export) |
| **Project** | `en.wikipedia` |
| **Article** | `2023 Cricket World Cup` |
| **Agent filter** | **user** (bots and automated traffic excluded) |
| **Granularity** | Daily |
| **Date range** | **1 October – 25 November 2023** (56 days) |
| **Raw file** | [`raw/pageviews-2023-10-01-2023-11-25.csv`](raw/pageviews-2023-10-01-2023-11-25.csv) |
| **Header** | `Date,2023 Cricket World Cup` |

### Retrieval Method

The daily page-view totals were exported manually from the Wikimedia Pageviews
Analysis tool at <https://pageviews.wmcloud.org/> by selecting the article
"2023 Cricket World Cup", project `en.wikipedia`, agent = **user**, daily
granularity, and the date range 1 Oct – 25 Nov 2023. The resulting CSV was
saved as `pageviews-2023-10-01-2023-11-25.csv` (56 data rows).

---

## 2. Build Method (Daily → Hourly)

The script [`build_hourly_workload.py`](build_hourly_workload.py) converts the
56 daily totals into 1,344 hourly values. For each day with real total **D**:

1. **Base profile:** 70 % of D is spread evenly over the match window
   14:00–21:59 (8 hours, 8.75 % each); 30 % over the other 16 hours
   (1.875 % each).
2. **Noise:** Each hour is multiplied by uniform random noise
   U(0.92, 1.08) — i.e. ±8 %.
3. **Renormalisation:** The 24 hours are rescaled so they sum to D again.
4. **Rounding:** Values are rounded to integers; the rounding remainder is
   added to the largest hour, so each day's sum equals D exactly.
5. **Seed:** Fixed seed 42 (`numpy.random.default_rng`), so the output is
   identical on every run.

---

## 3. Real vs. Synthetic Components

| Component | Status | Description |
|---|---|---|
| **Daily page-view totals** | **Real** (measured) | The 56 daily totals are empirical page-view counts from Wikimedia. They are preserved **exactly** in the hourly file (daily sum of hourly hits = daily total). |
| **Intra-day 70/30 split** | **Synthetic** (modelled) | 70 % allocated evenly over 8 match hours, 30 % over 16 off-peak hours. Flat distribution within each band. |
| **±8 % uniform noise** | **Synthetic** | Each hour multiplied by U(0.92, 1.08), then rescaled. |
| **Daily totals** | **Exact** | Renormalisation and integer rounding guarantee that the sum of 24 hourly values equals the real daily count. |

> **Note:** Only the daily totals represent real empirical measurements.
> The hourly profile and fluctuations are synthetically generated.
> This trace is **semi-synthetic** and should not be cited as empirical hourly server traffic.

---

## 4. Output Schema

| Column | Type | Example |
|---|---|---|
| `Timestamp` | `YYYY-MM-DD HH:MM:SS` | `2023-10-01 00:00:00` |
| `Hits` | integer | `759` |

- **Rows:** 1,344 (56 days × 24 hours)
- **First timestamp:** `2023-10-01 00:00:00`
- **Last timestamp:** `2023-11-25 23:00:00`
- **Total hits:** 29,957,762
- **Peak hour:** 117,612 at `2023-11-19 17:00:00` (hour index 1193)
- No trailing text, no junk lines.

---

## 5. Regeneration

```bash
python data/build_hourly_workload.py
```

With seed 42 (the default), this reproduces `cricket_hourly_workload.csv`
byte-identically. To verify:

```bash
python data/build_hourly_workload.py
git diff --stat data/cricket_hourly_workload.csv
# Expected: no output (no changes)
```

---

## 6. Why the Old Dataset Was Replaced

The previous hourly CSV was generated from 56 daily values that were typed in
by hand. Ten of those values were wrong:

- Four knockout match days (4, 15, 16, and 19 November) were capped at
  1,000,000 views.
- 5 November lost 600,000 views to a typo.
- Five other days had small transcription errors.

In addition, the old `build_hourly_workload.py` on GitHub implemented a
different algorithm (raised-cosine profile with log-normal noise) that was never
used to produce the committed results, and `data/raw/wikimedia_daily.csv` was
an all-agents download for a different date range (5 Oct – 30 Nov 2023).

The current version uses the correct daily totals exported directly from the
Wikimedia Pageviews tool with agent = user, and the build script matches the
algorithm that produced the canonical hourly trace.

---

## 7. Timezone Note

The 14:00–21:59 peak window was chosen to approximate local match hours in
India (IST, UTC+05:30) during the tournament. Wikimedia Pageviews daily
boundaries are in UTC. The generation script does not shift calendar day
boundaries; the intra-day profile is synthetic in either case.

---

## 8. Terms and Attribution

Wikimedia page view counts are made available under Wikimedia Terms of Use:
- [Wikimedia Terms of Use](https://wikimediafoundation.org/wiki/Terms_of_Use)
- [Wikimedia Pageviews API Documentation](https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews)
