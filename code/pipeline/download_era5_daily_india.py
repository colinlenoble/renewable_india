"""
Download ERA5 6-hourly GRIBs for India (1980-2020), aggregate to daily
resolution, then write one NetCDF per variable:
  era5_tas_1980_2020.nc
  era5_tasmax_1980_2020.nc
  era5_sfcWind_1980_2020.nc   (= sqrt(uas² + vas²))
  era5_rsds_1980_2020.nc

ssrd aggregation:
  The 6-hourly GRIB has shape (time=inits, step=2, lat, lon).
  Each value is a 1-hour J m-2 accumulation at the valid time.
  Sum 2 inits × 2 steps = 4 values per day → divide by 4×3600 → W m-2.

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
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]

VAR_FILES = {
    "tas":      OUTPUT_DIR / "era5_tas_1980_2020.nc",
    "tasmax":   OUTPUT_DIR / "era5_tasmax_1980_2020.nc",
    "sfcWind":  OUTPUT_DIR / "era5_sfcWind_1980_2020.nc",
    "rsds":     OUTPUT_DIR / "era5_rsds_1980_2020.nc",
}


def open_var(grib_path, short_name):
    ds = xr.open_dataset(
        str(grib_path), engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": short_name}},
    )
    da = ds[next(v for v in ds.data_vars if v != "valid_time")]
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    return da


def process_to_daily(grib_path, out_path):
    print("    processing → daily …")

    # Instantaneous: (time, lat, lon) with 4 values/day at 00/06/12/18 UTC
    t2m = open_var(grib_path, "2t")
    u10 = open_var(grib_path, "10u")
    v10 = open_var(grib_path, "10v")

    # Forecast: (time=inits, step, lat, lon) – 2 inits × 2 steps per day
    mx  = open_var(grib_path, "mx2t")
    ss  = open_var(grib_path, "ssrd")

    # ssrd: sum 4 1-hour J m-2 accumulations per day → W m-2
    rsds = (ss.resample(time="1D").sum().sum(dim="step") / (4 * 3600)).clip(min=0.0)

    ds = xr.Dataset({
        "tas":    t2m.resample(time="1D").mean(),
        "tasmax": mx.resample(time="1D").max().max(dim="step"),
        "uas":    u10.resample(time="1D").mean(),
        "vas":    v10.resample(time="1D").mean(),
        "rsds":   rsds,
    })

    for v in ds.data_vars:
        ds[v] = ds[v].astype("float32")
    ds.attrs = {"source": f"ERA5 6-hourly → daily, year {grib_path.stem}"}
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)


# ── Download and process each year ───────────────────────────────────────────
if all(p.exists() for p in VAR_FILES.values()):
    print("All per-variable files already exist — nothing to do.")
else:
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

    # ── Write per-variable files ──────────────────────────────────────────────
    yearly = sorted(OUTPUT_DIR.glob("era5_india_daily_*.nc"))
    if yearly:
        print(f"\nBuilding per-variable files from {len(yearly)} yearly files …")
        ds = xr.open_mfdataset(yearly, combine="by_coords")

        UNITS = {"tas": "K", "tasmax": "K", "sfcWind": "m s-1", "rsds": "W m-2"}
        enc   = {"zlib": True, "complevel": 4, "dtype": "float32"}

        VARS = {
            "tas":     ds["tas"],
            "tasmax":  ds["tasmax"],
            "sfcWind": np.hypot(ds["uas"], ds["vas"]).rename("sfcWind"),
            "rsds":    ds["rsds"],
        }

        for vname, da in VARS.items():
            out = VAR_FILES[vname]
            if out.exists():
                print(f"  {out.name}: exists — skipping")
                continue
            print(f"  Writing {out.name} …")
            da.attrs["units"] = UNITS[vname]
            da.to_dataset().to_netcdf(out, encoding={vname: enc})
            print(f"  done ({out.stat().st_size / 1e6:.0f} MB)")

        ds.close()

        for f in yearly:
            f.unlink()
        print("Per-year files deleted.")

print("All done.")
