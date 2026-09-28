#!/usr/bin/env python3
"""
Check whether the similar tas / sfcWind diurnal shapes in the downscaled hourly
output (historical) reflect a real physical coupling (daytime boundary-layer
mixing warms the surface AND brings higher-momentum air down, raising both tas
and sfcWind together; both decouple again at night) or an artifact of the
downscaling pipeline.

Diurnal library is unavailable right now, so this works directly off the
downscaled hourly output (already validated in verify_hourly_downscaled.py).
Decomposition mirrors hourly_profiles() in downscale_hourly.py exactly (same
reshape, same anomaly/ratio definitions) — the file is known strictly hourly
(n_days x 24), confirmed in verify_hourly_downscaled.py.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(r"E:/temp/GFDL-ESM4")
OUT_PNG  = Path("figures/check_tas_wind_link.png")

print("Loading historical hourly file …", flush=True)
ds = xr.open_dataset(DATA_DIR / "GFDL-ESM4_historical_hourly.nc", engine="h5netcdf")

n_lat, n_lon = ds.sizes["lat"], ds.sizes["lon"]
n_time = ds.sizes["time"]
n_days = n_time // 24
assert n_days * 24 == n_time, "expected strictly hourly, n_days*24 time steps"

tas_v  = ds["tas"].values.reshape(n_days, 24, n_lat, n_lon)
wind_v = ds["sfcWind"].values.reshape(n_days, 24, n_lat, n_lon)

tas_d  = tas_v.mean(axis=1)                                   # (n_days, lat, lon)
wind_d = wind_v.mean(axis=1)

tas_anom   = tas_v - tas_d[:, np.newaxis, :, :]                # K, mean 0 per day/cell
wind_ratio = np.where(
    wind_d[:, np.newaxis, :, :] > 0,
    wind_v / wind_d[:, np.newaxis, :, :],
    1.0,
)                                                                # ratio, mean ~1 per day/cell

print("Computing mean diurnal shapes …", flush=True)
tas_shape  = tas_anom.mean(axis=(0, 2, 3))          # (24,)
wind_shape = (wind_ratio - 1.0).mean(axis=(0, 2, 3))  # (24,)


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


tas_norm  = minmax(tas_shape)
wind_norm = minmax(wind_shape)
r_shape = np.corrcoef(tas_shape, wind_shape)[0, 1]
print(f"Domain-mean diurnal shape correlation (tas anomaly vs sfcWind ratio-1): r = {r_shape:.3f}")

print("Computing per-cell correlation across the full record …", flush=True)
ta = tas_anom.reshape(n_days * 24, n_lat, n_lon)
wr = (wind_ratio - 1.0).reshape(n_days * 24, n_lat, n_lon)

ta_c = ta - ta.mean(axis=0, keepdims=True)
wr_c = wr - wr.mean(axis=0, keepdims=True)
num = (ta_c * wr_c).sum(axis=0)
den = np.sqrt((ta_c**2).sum(axis=0) * (wr_c**2).sum(axis=0))
corr_map = np.where(den > 0, num / den, np.nan)

print(f"Per-cell correlation: mean={np.nanmean(corr_map):.3f}  "
      f"min={np.nanmin(corr_map):.3f}  max={np.nanmax(corr_map):.3f}  "
      f"frac positive={np.nanmean(corr_map > 0):.2f}")

# ── Figure ───────────────────────────────────────────────────────────────────
print("Building figure …", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.axhline(0, color="#999999", lw=1, ls="--")
ax.plot(range(24), tas_norm, color="#c0392b", lw=2, label="tas anomaly (norm.)")
ax.plot(range(24), wind_norm, color="#2471a3", lw=2, label="sfcWind ratio-1 (norm.)")
ax.set_xlabel("hour (UTC)")
ax.set_ylabel("normalised shape (0-1)")
ax.set_title(f"Mean diurnal shape — historical\nshape correlation r = {r_shape:.2f}")
ax.legend()

ax = axes[1]
vmax = np.nanmax(np.abs(corr_map))
im = ax.pcolormesh(ds["lon"], ds["lat"], corr_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
ax.set_title("Per-cell correlation:\nintraday tas anomaly vs sfcWind ratio-1")
ax.set_xlabel("lon")
ax.set_ylabel("lat")
fig.colorbar(im, ax=ax, label="Pearson r")

fig.suptitle("Is the tas/sfcWind diurnal similarity real or an artifact? (GFDL-ESM4 historical)")
fig.tight_layout()
OUT_PNG.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved figure -> {OUT_PNG}")
