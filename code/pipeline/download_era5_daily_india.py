"""
Download ERA5 6-hourly GRIBs for India (1980-2020), aggregate to daily
NetCDF immediately, then delete the GRIB to save disk space.

For each year the following daily variables are written to
  data/raw/era5_daily/era5_india_daily_{year}.nc:
  tas     = mean(t2m)   [K]
  tasmax  = max(mx2t at 06/18 UTC)  [K]
  tasmin  = min(mn2t at 06/18 UTC)  [K]
  uas     = mean(u10)   [m/s]
  vas     = mean(v10)   [m/s]
  rsds    = mean ssrd   [W/m²]

Final merged file: data/raw/era5_daily/era5_daily_1980_2020.nc

Run with: conda run -n xr_env python code/download_era5_daily_india.py
"""

import cdsapi
import numpy as np
import xarray as xr
from pathlib import Path

OUTPUT_DIR = Path("data/raw/era5_daily")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AREA      = [37, 68, 6, 97]   # [N, W, S, E]
MONTHS    = [f"{m:02d}" for m in range(1, 13)]
DAYS      = [f"{d:02d}" for d in range(1, 32)]
TIMES     = ["00:00", "06:00", "12:00", "18:00"]
VARIABLES = [
    "2m_temperature",
    "maximum_2m_temperature_since_previous_post_processing",
    "minimum_2m_temperature_since_previous_post_processing",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]


def load_var(grib_path, short_name, vname):
    ds = xr.open_dataset(
        str(grib_path), engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": short_name}},
    )
    actual = [v for v in ds.data_vars if v != "valid_time"]
    da = ds[actual[0]]
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"

    if "step" in da.dims:
        # Forecast variable: 4D (time, step, lat, lon) — flatten via valid_time
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
    ds = xr.open_dataset(
        str(grib_path), engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "ssrd"}},
    )
    ssrd = ds["ssrd"].load().fillna(0.0)
    lat_name = "latitude" if "latitude" in ssrd.dims else "lat"
    lon_name = "longitude" if "longitude" in ssrd.dims else "lon"
    hourly = (ssrd / (6 * 3600)).clip(min=0.0)

    if "step" in ssrd.dims:
        # Forecast format: 4D (time, step, lat, lon) — flatten using valid_time
        valid_time = ds["valid_time"].load()  # shape (time, step)
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
        # Already 3D (time, lat, lon)
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


def process_to_daily(grib_path, out_path):
    print(f"    processing → daily …")
    t2m  = load_var(grib_path, "2t",   "tas")
    mx2t = load_var(grib_path, "mx2t", "tasmax")
    mn2t = load_var(grib_path, "mn2t", "tasmin")
    u10  = load_var(grib_path, "10u",  "uas")
    v10  = load_var(grib_path, "10v",  "vas")
    ssrd = load_ssrd(grib_path)

    def daily_mean(da):
        return da.resample(time="1D").mean()

    def daily_max(da):
        h06 = da.sel(time=da.time.dt.hour == 6).assign_coords(
            time=lambda x: x.time.dt.floor("D"))
        h18 = da.sel(time=da.time.dt.hour == 18).assign_coords(
            time=lambda x: x.time.dt.floor("D"))
        return xr.concat([h06, h18], dim="sample").max("sample")

    def daily_min(da):
        h06 = da.sel(time=da.time.dt.hour == 6).assign_coords(
            time=lambda x: x.time.dt.floor("D"))
        h18 = da.sel(time=da.time.dt.hour == 18).assign_coords(
            time=lambda x: x.time.dt.floor("D"))
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
    ds.attrs = {"source": f"ERA5 6-hourly → daily, year {grib_path.stem}"}
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)


c = cdsapi.Client()

for year in range(1980, 2021):
    daily_nc = OUTPUT_DIR / f"era5_india_daily_{year}.nc"
    grib     = OUTPUT_DIR / f"era5_india_{year}_6h.grib"

    if daily_nc.exists():
        print(f"{year}: already processed — skipping")
        continue

    if not grib.exists():
        print(f"{year}: downloading …")
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type":    "reanalysis",
                    "variable":        VARIABLES,
                    "year":            str(year),
                    "month":           MONTHS,
                    "day":             DAYS,
                    "time":            TIMES,
                    "area":            AREA,
                    "data_format":     "grib",
                    "download_format": "unarchived",
                },
                str(grib),
            )
        except PermissionError:
            if not grib.exists():
                raise
            print(f"  (existing GRIB found after PermissionError — using it)")

    process_to_daily(grib, daily_nc)
    grib.unlink()
    print(f"{year}: → {daily_nc.name}  (GRIB deleted)")

# ── Merge all yearly files into one ──────────────────────────
merged = OUTPUT_DIR / "era5_daily_1980_2020.nc"
if not merged.exists():
    files = sorted(OUTPUT_DIR.glob("era5_india_daily_*.nc"))
    if files:
        print(f"\nMerging {len(files)} yearly files → {merged.name} …")
        xr.open_mfdataset(files, combine="by_coords").to_netcdf(merged)
        print(f"Done → {merged.name}")
else:
    print(f"\nAlready merged: {merged.name}")

print("All done.")
