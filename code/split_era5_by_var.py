"""
Split era5_daily_1980_2020.nc into one file per variable:
  era5_tas_1980_2020.nc
  era5_tasmax_1980_2020.nc
  era5_sfcWind_1980_2020.nc   (= sqrt(uas² + vas²))
  era5_rsds_1980_2020.nc

Then delete the per-year files and the merged file.

Run with: C:/Users/colin/anaconda3/envs/xr_env/python.exe code/split_era5_by_var.py
"""

import numpy as np
import xarray as xr
from pathlib import Path

ERA5_DIR = Path("data/raw/era5_daily")
merged   = ERA5_DIR / "era5_daily_1980_2020.nc"
enc      = {"zlib": True, "complevel": 4, "dtype": "float32"}

print("Opening merged file…", flush=True)
ds = xr.open_dataset(merged)
print(f"  dims: {dict(ds.sizes)}, vars: {list(ds.data_vars)}", flush=True)

VARS = {
    "tas":      ds["tas"],
    "tasmax":   ds["tasmax"],
    "sfcWind":  np.hypot(ds["uas"], ds["vas"]).rename("sfcWind"),
    "rsds":     ds["rsds"],
}

for vname, da in VARS.items():
    out = ERA5_DIR / f"era5_{vname}_1980_2020.nc"
    if out.exists():
        print(f"  {out.name}: exists — skipping", flush=True)
        continue
    print(f"  Writing {out.name} …", flush=True)
    da.attrs["units"] = {"tas": "K", "tasmax": "K", "sfcWind": "m s-1", "rsds": "W m-2"}[vname]
    da.to_dataset().to_netcdf(out, encoding={vname: enc})
    print(f"  done: {out.stat().st_size / 1e6:.0f} MB", flush=True)

ds.close()
print("Per-variable files written.", flush=True)

# Delete per-year files
yearly = sorted(ERA5_DIR.glob("era5_india_daily_*.nc"))
print(f"\nDeleting {len(yearly)} yearly files…", flush=True)
for f in yearly:
    f.unlink()
print("Deleted yearly files.", flush=True)

# Delete merged file
merged.unlink()
print(f"Deleted {merged.name}.", flush=True)

print("Done.", flush=True)
