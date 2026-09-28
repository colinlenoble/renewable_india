#!/usr/bin/env python3
"""
Single-pixel check: how often does daily-mean rsds drop so low there's
essentially no usable solar potential that day (deep, persistent cloud cover)?

Pixel: nearest grid cell to (lat=21N, lon=79E) — central India.
Threshold: 5th percentile of the HISTORICAL daily-mean rsds distribution at
that pixel, applied consistently to ssp126/ssp370 for comparability.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR   = Path(r"E:/temp/GFDL-ESM4")
OUT_PNG    = Path("figures/pixel_no_potential_rsds.png")
TARGET_LAT = 21.0
TARGET_LON = 79.0

SCENARIOS = ["historical", "ssp126", "ssp370"]
COLORS    = {"historical": "#2471a3", "ssp126": "#27ae60", "ssp370": "#c0392b"}

daily_series = {}
pixel_lat = pixel_lon = None

for scn in SCENARIOS:
    print(f"Loading {scn} …", flush=True)
    ds = xr.open_dataset(DATA_DIR / f"GFDL-ESM4_{scn}_hourly.nc", engine="h5netcdf")

    if pixel_lat is None:
        lat_idx = int(np.abs(ds["lat"].values - TARGET_LAT).argmin())
        lon_idx = int(np.abs(ds["lon"].values - TARGET_LON).argmin())
        pixel_lat = float(ds["lat"].values[lat_idx])
        pixel_lon = float(ds["lon"].values[lon_idx])
        print(f"  Pixel selected: lat={pixel_lat}, lon={pixel_lon}")

    n_lat, n_lon = ds.sizes["lat"], ds.sizes["lon"]
    n_time = ds.sizes["time"]
    n_days = n_time // 24
    assert n_days * 24 == n_time

    rsds_v = ds["rsds"].values.reshape(n_days, 24, n_lat, n_lon)
    pixel_daily = rsds_v[:, :, lat_idx, lon_idx].mean(axis=1)   # (n_days,)
    dates = pd.date_range(start=pd.Timestamp(ds["time"].values[0]), periods=n_days, freq="D")
    daily_series[scn] = pd.Series(pixel_daily, index=dates)
    del ds, rsds_v

threshold = np.percentile(daily_series["historical"].values, 5)
print(f"\nThreshold (5th pct of historical daily-mean rsds at pixel): {threshold:.1f} W/m2")

low_days = {}
annual_counts = {}
for scn in SCENARIOS:
    s = daily_series[scn]
    flag = s < threshold
    low_days[scn] = s[flag]
    annual_counts[scn] = flag.groupby(flag.index.year).sum()
    print(f"{scn}: {flag.sum()} no-potential days out of {len(s)} "
          f"({100*flag.mean():.1f}%)")

# Longest consecutive run across all scenarios combined
best_run = None
for scn in SCENARIOS:
    flag = (daily_series[scn] < threshold).values
    idx = daily_series[scn].index
    run_start = None
    for i, v in enumerate(flag):
        if v and run_start is None:
            run_start = i
        if (not v or i == len(flag) - 1) and run_start is not None:
            run_end = i if not v else i + 1
            length = run_end - run_start
            if best_run is None or length > best_run[0]:
                best_run = (length, scn, idx[run_start], idx[run_end - 1])
            run_start = None

print(f"Longest run: {best_run[0]} consecutive days, {best_run[1]}, "
      f"{best_run[2].date()} -> {best_run[3].date()}")

# ── Figure ───────────────────────────────────────────────────────────────────
print("Building figure …", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
bins = np.linspace(0, 500, 60)
for scn in SCENARIOS:
    ax.hist(daily_series[scn].values, bins=bins, histtype="step", lw=1.8,
             color=COLORS[scn], label=scn, density=True)
ax.axvline(threshold, color="#333333", ls="--", lw=1.5, label=f"threshold={threshold:.0f}")
ax.set_xlabel("daily-mean rsds (W m$^{-2}$)")
ax.set_ylabel("density")
ax.set_title(f"Distribution at pixel ({pixel_lat:.2f}N, {pixel_lon:.2f}E)")
ax.legend(fontsize=8)

ax = axes[0, 1]
for scn in SCENARIOS:
    ac = annual_counts[scn]
    ax.plot(ac.index, ac.values, color=COLORS[scn], lw=1, alpha=0.5)
    ax.plot(ac.index, ac.rolling(10, center=True).mean(), color=COLORS[scn], lw=2.2, label=scn)
ax.set_xlabel("year")
ax.set_ylabel("no-potential days / year")
ax.set_title("Annual count (thin=raw, thick=10-yr rolling mean)")
ax.legend(fontsize=8)

ax = axes[1, 0]
length, scn, d0, d1 = best_run
window = daily_series[scn].loc[d0 - pd.Timedelta(days=2): d1 + pd.Timedelta(days=2)]
ax.plot(window.index, window.values, color=COLORS[scn], marker="o", lw=2)
ax.axhline(threshold, color="#333333", ls="--", lw=1)
ax.axvspan(d0, d1, color=COLORS[scn], alpha=0.15)
ax.set_title(f"Longest no-potential run: {length}d, {scn}\n{d0.date()} -> {d1.date()}")
ax.set_ylabel("daily-mean rsds (W m$^{-2}$)")
ax.tick_params(axis="x", rotation=30)

ax = axes[1, 1]
totals = [low_days[scn].shape[0] / len(daily_series[scn]) * 100 for scn in SCENARIOS]
ax.bar(SCENARIOS, totals, color=[COLORS[s] for s in SCENARIOS])
for i, v in enumerate(totals):
    ax.annotate(f"{v:.1f}%", (i, v), ha="center", va="bottom")
ax.set_ylabel("% of days below threshold")
ax.set_title("Overall no-potential day frequency")

fig.suptitle(f"GFDL-ESM4 — single-pixel solar 'no potential' days ({pixel_lat:.2f}N, {pixel_lon:.2f}E)")
fig.tight_layout()
OUT_PNG.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved figure -> {OUT_PNG}")
