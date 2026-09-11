"""
Synthetic workload generator.
Recreates the 4 workload patterns from Fig. 1 of Nashaat et al. (2026):
  (a) Uniform / bell-shaped (rises to a peak, then falls)
  (b) Linear, low change rate (~0.35 req/s per minute)
  (c) Linear, high change rate (~2.31 req/s per minute)
  (d) Random (unpredictable)

Each generator returns a pandas DataFrame with columns:
  minute            -> int, time index in minutes (matches Shashank's expected
                        monitoring format: one row per minute)
  avg_requests_per_min -> float, average requests during that minute
"""

import numpy as np
import pandas as pd


def _to_df(minutes, values):
    return pd.DataFrame({"minute": minutes, "avg_requests_per_min": values})


def generate_uniform_workload(duration_min=697, peak=300, noise_std=2.0, seed=42):
    """Bell-shaped workload: ramps up to `peak` around the midpoint, then ramps
    back down. Mirrors Fig. 1a."""
    rng = np.random.default_rng(seed)
    minutes = np.arange(duration_min)
    mid = duration_min // 2
    ramp_up = np.linspace(0, peak, mid)
    ramp_down = np.linspace(peak, 0, duration_min - mid)
    base = np.concatenate([ramp_up, ramp_down])
    noise = rng.normal(0, noise_std, duration_min)
    values = np.clip(base + noise, 0, None)
    return _to_df(minutes, values)


def generate_linear_workload(duration_min=697, rate=0.35, noise_std=1.0, seed=42):
    """Steady linear growth at `rate` requests/sec per minute.
    rate=0.35 -> Fig. 1b (low rate), rate=2.31 -> Fig. 1c (high rate)."""
    rng = np.random.default_rng(seed)
    minutes = np.arange(duration_min)
    base = rate * minutes
    noise = rng.normal(0, noise_std, duration_min)
    values = np.clip(base + noise, 0, None)
    return _to_df(minutes, values)


def generate_random_workload(duration_min=697, base_level=200, volatility=80, seed=42):
    """Unpredictable workload with sharp ups/downs. Mirrors Fig. 1d.
    Built as a bounded random walk so it stays somewhat realistic (no
    negative requests, no runaway drift)."""
    rng = np.random.default_rng(seed)
    minutes = np.arange(duration_min)
    steps = rng.normal(0, volatility / 10, duration_min)
    walk = np.cumsum(steps)
    # re-center and clip so it oscillates around base_level like Fig 1d
    walk = walk - walk.mean()
    values = np.clip(base_level + walk, 0, None)
    return _to_df(minutes, values)


PATTERNS = {
    "uniform": generate_uniform_workload,
    "linear_low": lambda **kw: generate_linear_workload(rate=0.35, **kw),
    "linear_high": lambda **kw: generate_linear_workload(rate=2.31, **kw),
    "random": generate_random_workload,
}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (name, fn) in zip(axes.flat, PATTERNS.items()):
        df = fn(duration_min=697)
        ax.plot(df["minute"], df["avg_requests_per_min"])
        ax.set_title(name)
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Avg requests/min")
    plt.tight_layout()
    plt.savefig("workload_patterns_preview.png", dpi=100)
    print("Saved preview to workload_patterns_preview.png")
    for name, fn in PATTERNS.items():
        df = fn(duration_min=697)
        print(name, df["avg_requests_per_min"].describe().to_dict())
