"""
Fix existing era5_india_daily_*.nc files that were saved with a spurious
'step' dimension on tasmax/tasmin. Reconstructs valid_time from init+step,
re-computes daily max/min correctly, and re-saves with zlib compression.

Run with: conda run -n xr_env python code/fix_era5_daily_step_dim.py
"""

import numpy as np
import xarray as xr
from pathlib import Path

INPUT_DIR = Path("data/raw/era5_daily")


def flatten_step(da):
    """Flatten (time=init, step, lat, lon) → (valid_time, lat, lon)."""
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"
    n_lat = da.sizes[lat_name]
    n_lon = da.sizes[lon_name]

    init_times = da.time.values          # shape (n_init,)
    steps = da.step.values               # shape (n_step,) — numpy timedelta64

    # valid_time[i,j] = init_times[i] + steps[j]
    flat_valid = np.array([
        t + s for t in init_times for s in steps
    ], dtype="datetime64[ns]")
    flat_data = da.values.reshape(-1, n_lat, n_lon)

    sort_idx = np.argsort(flat_valid)
    flat_valid = flat_valid[sort_idx]
    flat_data  = flat_data[sort_idx]

    # Drop duplicate valid times (can arise when steps overlap across inits)
    _, unique_idx = np.unique(flat_valid, return_index=True)
    flat_valid = flat_valid[unique_idx]
    flat_data  = flat_data[unique_idx]

    return xr.DataArray(
        flat_data,
        dims=["time", lat_name, lon_name],
        coords={"time": flat_valid,
                lat_name: da[lat_name].values,
                lon_name: da[lon_name].values},
    )


def daily_max(da):
    h06 = da.sel(time=da.time.dt.hour == 6)
    h18 = da.sel(time=da.time.dt.hour == 18)
    h06 = h06.assign_coords(time=h06.time.dt.floor("D"))
    h18 = h18.assign_coords(time=h18.time.dt.floor("D"))
    return xr.concat([h06, h18], dim="sample").max("sample")


def daily_min(da):
    h06 = da.sel(time=da.time.dt.hour == 6)
    h18 = da.sel(time=da.time.dt.hour == 18)
    h06 = h06.assign_coords(time=h06.time.dt.floor("D"))
    h18 = h18.assign_coords(time=h18.time.dt.floor("D"))
    return xr.concat([h06, h18], dim="sample").min("sample")


files = sorted(INPUT_DIR.glob("era5_india_daily_*.nc"))
enc = {v: {"zlib": True, "complevel": 4} for v in
       ["tas", "tasmax", "tasmin", "uas", "vas", "rsds"]}

for f in files:
    ds = xr.open_dataset(f)

    has_step = any("step" in ds[v].dims for v in ds.data_vars)
    if not has_step:
        print(f"{f.name}: no step dim — skipping")
        ds.close()
        continue

    print(f"{f.name}: recomputing …", flush=True)

    fixed = {}
    for v in ds.data_vars:
        da = ds[v].load()
        if "step" in da.dims:
            da = flatten_step(da)
            if v == "tasmax":
                da = daily_max(da)
            elif v == "tasmin":
                da = daily_min(da)
            # other vars with step (shouldn't happen) → just take mean
            else:
                da = da.resample(time="1D").mean()
        fixed[v] = da.astype("float32")

    ds_fixed = xr.Dataset(fixed, attrs=ds.attrs)
    ds.close()

    tmp = f.with_suffix(".tmp.nc")
    ds_fixed.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4}
                                      for v in ds_fixed.data_vars})
    tmp.replace(f)
    print(f"  → {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")

print("All fixed.")
