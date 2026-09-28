"""
Download ERA5 hourly GRIB for India (2019) and convert to NetCDF.

Output: data/raw/era5/era5_india_2019_hourly.nc
Variables:
  tas      = 2m temperature             [K]
  sfcWind  = 10m wind speed             [m s-1]  (= sqrt(u10²+v10²))
  rsds     = surface solar radiation    [W m-2]

ssrd conversion:
  Hourly ERA5 ssrd is cumulative from each forecast init (06 and 18 UTC).
  Deaccumulate with np.diff along the step axis, then divide by 3600 → W m-2.

Run with: conda run -n xr_env python code/download_era5_hourly.py
"""

import cdsapi
import numpy as np
import xarray as xr
from pathlib import Path

OUTPUT_DIR = Path("data/raw/era5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR    = 2019
GRIB    = OUTPUT_DIR / f"era5_india_{YEAR}_hourly.grib"
NC_OUT  = OUTPUT_DIR / f"era5_india_{YEAR}_hourly.nc"

AREA    = [37, 68, 6, 97]   # [N, W, S, E]
MONTHS  = [f"{m:02d}" for m in range(1, 13)]
DAYS    = [f"{d:02d}" for d in range(1, 32)]
TIMES   = [f"{h:02d}:00" for h in range(24)]
VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]


def open_inst(grib_path, short_name):
    """Load an instantaneous variable; returns (time, lat, lon) DataArray."""
    ds = xr.open_dataset(
        str(grib_path), engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": short_name}},
    )
    da = ds[next(v for v in ds.data_vars if v != "valid_time")]
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    return da

def load_ssrd_hourly(grib_path):
    """
    Load ssrd from hourly GRIB.
    ERA5 hourly ssrd is cumulative from each forecast init (06/18 UTC).
    Deaccumulate with np.diff along the step axis, flatten to (time, lat, lon).
    """
    ds = xr.open_dataset(
        str(grib_path), engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "ssrd"}},
    )
    ds = xr.open_dataset(
        str(grib_path), engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "ssrd"}},
    )
    ds = ds.fillna(0)
    ds['time_step'] = ds['time'] + ds['step']
    ds = ds.stack(time_dim=('time', 'step')).reset_index('time_dim')
    ds['time_dim'] = ds['time'] + ds['step']
    ds['ssrd'] = ds['ssrd'] / (3600)  # convert from J m-2 to W m-2
    #drop time and step
    ds = ds.drop_vars(['time', 'step'])
    ds = ds.rename({'time_dim': 'time', 'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.drop_vars(['surface', 'number', 'valid_time', 'time_step'])
    ds = ds.rename({'ssrd': 'rsds'})
    ds = ds.sel(time = slice(str(YEAR) + '-01-01', str(YEAR) + '-12-31'))

    return ds['rsds']

def convert_to_nc(grib_path, nc_path):
    print("Converting GRIB → NetCDF …")

    t2m = open_inst(grib_path, "2t")
    u10 = open_inst(grib_path, "10u")
    v10 = open_inst(grib_path, "10v")
    rsds = load_ssrd_hourly(grib_path)

    sfcWind = np.hypot(u10, v10).rename("sfcWind")
    sfcWind.attrs["units"] = "m s-1"

    tas = t2m.rename("tas")
    tas.attrs["units"] = "K"


    # Align all variables to the rsds time axis (which may differ slightly
    # at day boundaries due to init-time grouping)
    ds = xr.Dataset({"tas": tas, "sfcWind": sfcWind, "rsds": rsds})

    enc = {"zlib": True, "complevel": 4, "dtype": "float32"}
    ds.to_netcdf(nc_path, encoding={v: enc for v in ds.data_vars})
    print(f"Saved → {nc_path}  ({nc_path.stat().st_size / 1e6:.0f} MB)")


# ── Download ──────────────────────────────────────────────────────────────────
if NC_OUT.exists():
    print(f"Already exists: {NC_OUT} — skipping.")
else:
    if not GRIB.exists():
        print(f"Downloading ERA5 hourly {YEAR} …")
        c = cdsapi.Client()
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type":    "reanalysis",
                    "variable":        VARIABLES,
                    "year":            str(YEAR),
                    "month":           MONTHS,
                    "day":             DAYS,
                    "time":            TIMES,
                    "area":            AREA,
                    "data_format":     "grib",
                    "download_format": "unarchived",
                },
                str(GRIB),
            )
        except PermissionError:
            if not GRIB.exists():
                raise
            print("  (existing GRIB found after PermissionError — using it)")
        print(f"Downloaded → {GRIB}  ({GRIB.stat().st_size / 1e6:.0f} MB)")

    convert_to_nc(GRIB, NC_OUT)
    GRIB.unlink()
    print("GRIB deleted.")

print("Done.")
