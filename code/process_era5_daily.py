"""
Aggregate 6-hourly ERA5 GRIBs to daily NetCDF files.
Run after download_era5_daily_india.py.

For each year GRIB the following daily variables are computed:
  tas     = mean(t2m at 00, 06, 12, 18 UTC)          [K]
  tasmax  = max(mx2t at 06 UTC, mx2t at 18 UTC)       [K]
  tasmin  = min(mn2t at 06 UTC, mn2t at 18 UTC)       [K]
  uas     = mean(u10 at 00, 06, 12, 18 UTC)           [m/s]
  vas     = mean(v10 at 00, 06, 12, 18 UTC)           [m/s]
  rsds    = daily mean ssrd [W/m²] via deaccumulation  [W/m²]

Output: data/raw/era5_daily/era5_india_daily_{year}.nc
Final merged: data/raw/era5_daily/era5_daily_1980_2020.nc

Run with: conda run -n xr_env python code/process_era5_daily.py
"""

import numpy as np
import xarray as xr
from pathlib import Path

INPUT_DIR  = Path("data/raw/era5_daily")
YEARS      = range(1980, 2021)


def load_var(grib_path, short_name, vname):
    """Load a variable from the 6-hourly GRIB, flattening step dim if needed."""
    ds = xr.open_dataset(
        str(grib_path), engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": short_name}},
    )
    actual = [v for v in ds.data_vars if v != "valid_time"]
    da = ds[actual[0]]
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"

    if "step" in da.dims:
        valid_time = ds["valid_time"].load()
        n_lat, n_lon = da.sizes[lat_name], da.sizes[lon_name]
        flat_time = valid_time.values.flatten()
        flat_data = da.values.reshape(-1, n_lat, n_lon)
        sort_idx = np.argsort(flat_time)
        flat_time, flat_data = flat_time[sort_idx], flat_data[sort_idx]
        _, unique_idx = np.unique(flat_time, return_index=True)
        flat_time, flat_data = flat_time[unique_idx], flat_data[unique_idx]
        da = xr.DataArray(
            flat_data, dims=["time", lat_name, lon_name],
            coords={"time": flat_time, lat_name: da[lat_name].values, lon_name: da[lon_name].values},
        )

    da = da.rename(vname)
    if lat_name == "latitude":
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    return da


def load_ssrd(grib_path):
    """Load ssrd; convert J/m² per 6-hour step → W/m²."""
    ds = xr.open_dataset(
        str(grib_path), engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "ssrd"}},
    )
    ssrd = ds["ssrd"].load().fillna(0.0)
    lat_name = "latitude" if "latitude" in ssrd.dims else "lat"
    lon_name = "longitude" if "longitude" in ssrd.dims else "lon"
    hourly = (ssrd / (6 * 3600)).clip(min=0.0)

    if "step" in ssrd.dims:
        valid_time = ds["valid_time"].load()
        n_lat = ssrd.sizes[lat_name]
        n_lon = ssrd.sizes[lon_name]
        flat_time = valid_time.values.flatten()
        flat_data = hourly.values.reshape(-1, n_lat, n_lon)
        sort_idx = np.argsort(flat_time)
        flat_time = flat_time[sort_idx]
        flat_data = flat_data[sort_idx]
        _, unique_idx = np.unique(flat_time, return_index=True)
        flat_time = flat_time[unique_idx]
        flat_data = flat_data[unique_idx]
    else:
        flat_time = ssrd.time.values
        flat_data = hourly.values

    da = xr.DataArray(
        flat_data,
        dims=["time", lat_name, lon_name],
        coords={"time": flat_time, lat_name: ssrd[lat_name].values, lon_name: ssrd[lon_name].values},
        name="rsds",
    )
    if lat_name == "latitude":
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    return da.sortby("time")


def process_year(year):
    grib = INPUT_DIR / f"era5_india_{year}_6h.grib"
    out  = INPUT_DIR / f"era5_india_daily_{year}.nc"

    if out.exists():
        print(f"  {year}: already processed — skipping")
        return

    if not grib.exists():
        print(f"  {year}: GRIB not found — skipping")
        return

    print(f"  {year}: loading …")

    t2m  = load_var(grib, "2t",    "tas")
    mx2t = load_var(grib, "mx2t",  "tasmax")
    mn2t = load_var(grib, "mn2t",  "tasmin")
    u10  = load_var(grib, "10u",   "uas")
    v10  = load_var(grib, "10v",   "vas")
    ssrd = load_ssrd(grib)

    print(f"  {year}: computing daily stats …")

    def daily_mean(da):
        return da.resample(time="1D").mean()

    def daily_max(da):
        # mx2t resets at 06 and 18 UTC → take max of the two daily values
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

    ds = xr.Dataset({
        "tas":    daily_mean(t2m),
        "tasmax": daily_max(mx2t),
        "tasmin": daily_min(mn2t),
        "uas":    daily_mean(u10),
        "vas":    daily_mean(v10),
        "rsds":   daily_mean(ssrd),
    })

    for v in ds.data_vars:
        ds[v] = ds[v].astype("float32")

    ds.attrs = {"source": f"ERA5 6-hourly → daily, year {year}"}
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out, encoding=encoding)
    print(f"  {year}: → {out.name}")


# ── Process all years ────────────────────────────────────────
print("Processing yearly GRIBs to daily NetCDF …")
for year in YEARS:
    process_year(year)

# ── Merge into single file ───────────────────────────────────
merged = INPUT_DIR / "era5_daily_1980_2020.nc"
if not merged.exists():
    files = sorted(INPUT_DIR.glob("era5_india_daily_*.nc"))
    if files:
        print(f"\nMerging {len(files)} yearly files → {merged.name} …")
        xr.open_mfdataset(files, combine="by_coords").to_netcdf(merged)
        print(f"Done → {merged.name}")
    else:
        print("No yearly daily files found to merge.")
else:
    print(f"\nAlready merged: {merged.name}")
