"""
Merge per-year ERA5 daily files into combined NetCDF files.
Run after download_era5_daily_india.py completes.

Output (data/raw/era5_daily/):
  era5_daily_mean_1980_2020.nc   — tas, uas, vas, rsds  (all merged)
  era5_daily_max_1980_2020.nc    — tasmax
  era5_daily_min_1980_2020.nc    — tasmin

Run with: conda run -n xr_env python code/merge_era5_daily.py
"""

import xarray as xr
from pathlib import Path

OUTPUT_DIR = Path("data/raw/era5_daily")
YEARLY_DIR = OUTPUT_DIR / "yearly"

SHORT_NAMES = ["tas", "uas", "vas", "rsds", "tasmax", "tasmin"]

# ── 1. Merge each variable across years ─────────────────────
var_datasets = {}
for short in SHORT_NAMES:
    out = OUTPUT_DIR / f"era5_{short}_1980_2020.nc"
    if out.exists():
        print(f"Already merged: {out.name}")
        var_datasets[short] = xr.open_dataset(out)
        continue
    files = sorted(YEARLY_DIR.glob(f"era5_{short}_*.nc"))
    if not files:
        print(f"No yearly files for '{short}' — run download first")
        continue
    print(f"Merging {len(files)} files for {short} …")
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds.to_netcdf(out)
    print(f"  → {out.name}")
    var_datasets[short] = ds

# ── 2. Combine into grouped files ───────────────────────────
# Combined mean file (tas, uas, vas, rsds)
mean_out = OUTPUT_DIR / "era5_daily_mean_1980_2020.nc"
if not mean_out.exists() and all(s in var_datasets for s in ["tas", "uas", "vas", "rsds"]):
    print("Building combined mean file …")
    ds_mean = xr.merge([var_datasets[s] for s in ["tas", "uas", "vas", "rsds"]])
    ds_mean.to_netcdf(mean_out)
    print(f"  → {mean_out.name}")

for short, suffix in [("tasmax", "max"), ("tasmin", "min")]:
    combo = OUTPUT_DIR / f"era5_daily_{suffix}_1980_2020.nc"
    if not combo.exists() and short in var_datasets:
        var_datasets[short].to_netcdf(combo)
        print(f"  → {combo.name}")

print("Merge complete.")
