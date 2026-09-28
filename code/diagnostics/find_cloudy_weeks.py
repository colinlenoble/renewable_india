#!/usr/bin/env python3
"""
Find the weeks with the largest day-to-day swings in domain-mean daily rsds
for each SSP — a proxy for strong cloud-cover variability (e.g. a monsoon
system moving in/out, or a run of clear days broken by a storm).

Metric: 7-day centered rolling sum of |day-to-day difference| in domain-mean
daily rsds. Top-3 non-overlapping peak weeks are picked per scenario.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(r"E:/temp/GFDL-ESM4")
OUT_PNG  = Path("figures/cloudy_weeks_rsds.png")

SSPS   = ["ssp126", "ssp370"]
COLORS = {"ssp126": "#27ae60", "ssp370": "#c0392b"}
N_TOP  = 3

results = {}

for ssp in SSPS:
    print(f"Loading {ssp} …", flush=True)
    ds = xr.open_dataset(DATA_DIR / f"GFDL-ESM4_{ssp}_hourly.nc", engine="h5netcdf")

    n_lat, n_lon = ds.sizes["lat"], ds.sizes["lon"]
    n_time = ds.sizes["time"]
    n_days = n_time // 24
    assert n_days * 24 == n_time

    rsds_v = ds["rsds"].values.reshape(n_days, 24, n_lat, n_lon)
    domain_daily = rsds_v.mean(axis=(1, 2, 3))   # daily mean, then domain mean -> (n_days,)
    dates = pd.date_range(start=pd.Timestamp(ds["time"].values[0]), periods=n_days, freq="D")
    s = pd.Series(domain_daily, index=dates)

    dday = s.diff().abs()
    vol = dday.rolling(7, center=True, min_periods=7).sum()

    vol_work = vol.copy()
    top_weeks = []
    for _ in range(N_TOP):
        idx = vol_work.idxmax()
        if pd.isna(vol_work.loc[idx]):
            break
        top_weeks.append(idx)
        vol_work.loc[idx - pd.Timedelta(days=6): idx + pd.Timedelta(days=6)] = np.nan
    top_weeks = sorted(top_weeks)

    print(f"  Top {len(top_weeks)} volatile weeks (center date, 7-day sum |diff rsds|):")
    for idx in top_weeks:
        print(f"    {idx.date()}  score={vol.loc[idx]:.0f} W/m2")

    results[ssp] = dict(series=s, vol=vol, top_weeks=top_weeks)
    del ds, rsds_v

# ── Figure ───────────────────────────────────────────────────────────────────
print("Building figure …", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for col, ssp in enumerate(SSPS):
    r = results[ssp]
    color = COLORS[ssp]

    ax = axes[0, col]
    ax.plot(r["vol"].index, r["vol"].values, color=color, lw=0.8)
    for idx in r["top_weeks"]:
        ax.axvline(idx, color="#333333", lw=1, ls="--", alpha=0.7)
    ax.set_title(f"{ssp} — 7-day rolling Σ|Δ daily rsds| (domain mean)")
    ax.set_ylabel("W m$^{-2}$ per week")

    ax = axes[1, col]
    if r["top_weeks"]:
        idx0 = r["top_weeks"][0]
        window = r["series"].loc[idx0 - pd.Timedelta(days=3): idx0 + pd.Timedelta(days=3)]
        ax.plot(window.index, window.values, color=color, marker="o", lw=2)
        for d, v in window.items():
            ax.annotate(f"{v:.0f}", (d, v), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8)
        ax.set_title(f"{ssp} — most volatile week ({idx0.date()})")
    ax.set_ylabel("W m$^{-2}$")
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("GFDL-ESM4 — weeks with the largest day-to-day rsds swings (cloud-cover variability proxy)")
fig.tight_layout()
OUT_PNG.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved figure -> {OUT_PNG}")
