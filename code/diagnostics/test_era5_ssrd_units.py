#!/usr/bin/env python3
"""
Diagnostic: download one year of ERA5 6-hourly data, keep the raw GRIB,
and compare every plausible ssrd conversion against the existing hourly NC.

What we test
------------
The 6-hourly CDS download at times 00/06/12/18 UTC gives ssrd values that
are accumulated from the *forecast cycle init* (06 or 18 UTC), NOT per 6-h
step.  The two init times produce:

  18 UTC init → valid 00 UTC (step  6 h) → 6 h accumulation  ✓  / (6*3600)
  18 UTC init → valid 06 UTC (step 12 h) → 12 h accumulation ✗  was /  (6*3600)
  06 UTC init → valid 12 UTC (step  6 h) → 6 h accumulation  ✓  / (6*3600)
  06 UTC init → valid 18 UTC (step 12 h) → 12 h accumulation ✗  was / (6*3600)

Correct approach: deaccumulate with np.diff(ssrd, prepend=0) per cycle,
then / 3600 → W m-2 per step.

Usage
-----
python code/test_era5_ssrd_units.py [--year 2020] [--hourly-nc data/proc/era5/era5_india_2025_hourly.nc]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import cdsapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

AREA   = [37, 68, 6, 97]   # [N, W, S, E] India
TIMES  = ["00:00", "06:00", "12:00", "18:00"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]


# ── download ──────────────────────────────────────────────────────────────────

# Candidate locations where a 6-hourly GRIB for a given year might already exist
# (from previous runs of this script or download_era5_daily_india.py).
_GRIB_CANDIDATES = [
    "data/raw/era5_test/era5_india_{year}_6h_ssrd_test.grib",
    "data/raw/era5_daily/era5_india_{year}_6h.grib",
    "data/raw/era5/era5_india_{year}_6h.grib",
]


def find_existing_grib(year: int) -> Path | None:
    """Return the first existing GRIB for *year* across candidate locations."""
    for pattern in _GRIB_CANDIDATES:
        p = Path(pattern.format(year=year))
        if p.exists():
            return p
    return None


def get_or_download_grib(year: int, out_path: Path) -> Path:
    """
    Return a Path to a 6-hourly GRIB for *year*.
    Checks all candidate locations before triggering a new download.
    """
    existing = find_existing_grib(year)
    if existing is not None:
        log.info("Found existing GRIB: %s — skipping download", existing)
        return existing

    log.info("No existing GRIB found for %d — downloading …", year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type":    "reanalysis",
            "variable":        ["surface_solar_radiation_downwards"],
            "year":            str(year),
            "month":           MONTHS,
            "day":             DAYS,
            "time":            TIMES,
            "area":            AREA,
            "data_format":     "grib",
            "download_format": "unarchived",
        },
        str(out_path),
    )
    log.info("Downloaded → %s  (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


# ── load raw GRIB ssrd ────────────────────────────────────────────────────────

def load_raw_ssrd(grib_path: Path) -> xr.DataArray:
    """
    Load ssrd from the 6-hourly GRIB exactly as cfgrib sees it.
    Returns a DataArray with original J m-2 values and a flat time axis
    corresponding to valid_time (sorted, deduplicated).
    """
    import cfgrib
    ds = cfgrib.open_dataset(
        str(grib_path),
        backend_kwargs={"filter_by_keys": {"shortName": "ssrd"}},
        indexpath=None,
    )
    ssrd = ds["ssrd"]
    valid_time = ds["valid_time"]

    log.info("Raw GRIB shape  : %s  dims=%s", ssrd.shape, dict(ssrd.sizes))
    log.info("Raw GRIB attrs  : %s", ssrd.attrs)
    log.info("ssrd min/mean/max (J m-2): %.0f / %.0f / %.0f",
             float(ssrd.min()), float(ssrd.mean()), float(ssrd.max()))

    # Flatten (init_time, step, lat, lon) → (time, lat, lon)
    lat = ds.latitude.values if "latitude" in ds.coords else ds.lat.values
    lon = ds.longitude.values if "longitude" in ds.coords else ds.lon.values
    n_lat, n_lon = len(lat), len(lon)

    vt_flat   = valid_time.values.flatten()
    raw_flat  = ssrd.values.reshape(-1, n_lat, n_lon)
    sort_idx  = np.argsort(vt_flat)
    vt_flat   = vt_flat[sort_idx]
    raw_flat  = raw_flat[sort_idx]
    _, unique_idx = np.unique(vt_flat, return_index=True)
    vt_flat  = vt_flat[unique_idx]
    raw_flat = raw_flat[unique_idx]

    return xr.DataArray(
        raw_flat.astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={"time": vt_flat, "lat": lat, "lon": lon},
        name="ssrd_raw_Jm2",
        attrs={"units": "J m-2", "note": "raw CDS accumulation from cycle init"},
    )


# ── conversion A: current (naive /6/3600) ────────────────────────────────────

def convert_naive(ssrd_raw: xr.DataArray) -> xr.DataArray:
    """Current approach in download_era5_daily_india.py: divide by 6*3600."""
    rsds = (ssrd_raw / (6 * 3600)).clip(min=0.0)
    rsds.name = "rsds_naive"
    rsds.attrs = {"units": "W m-2", "method": "raw / (6*3600) — INCORRECT for step-12 h values"}
    return rsds


# ── conversion B: correct deaccumulation per cycle ───────────────────────────

def convert_deaccum(ssrd_raw: xr.DataArray) -> xr.DataArray:
    """
    Correct approach: diff per cycle (same as convert_era5_hourly.py).
    The GRIB has shape (n_inits, n_steps, lat, lon).
    We reconstruct that structure from the flat sorted array and diff along
    the step axis, resetting at each cycle.
    """
    times = ssrd_raw.time.values
    lat   = ssrd_raw.lat.values
    lon   = ssrd_raw.lon.values

    # Group by cycle: ERA5 inits at 06 and 18 UTC.
    # In the 4-sample-per-day layout the pattern is:
    #   init 18:00 prev → step 6  (00 UTC)
    #   init 18:00 prev → step 12 (06 UTC)
    #   init 06:00 same → step 6  (12 UTC)
    #   init 06:00 same → step 12 (18 UTC)
    # So steps alternate [6,12,6,12,...].  Per-cycle diff:
    #   step 6  → ssrd_step6 - 0         (first step in cycle)
    #   step 12 → ssrd_step12 - ssrd_step6

    raw = ssrd_raw.values  # (T, lat, lon)
    deacc = np.empty_like(raw)
    # Each pair of consecutive samples belongs to the same cycle
    for i in range(0, len(times), 2):
        deacc[i]     = np.maximum(raw[i],          0)          # step-6 from init
        deacc[i + 1] = np.maximum(raw[i + 1] - raw[i], 0)      # step-12 minus step-6

    # Convert from J m-2 per 6 h to W m-2 (mean over 6 h)
    rsds = xr.DataArray(
        (deacc / (6 * 3600)).astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lat, "lon": lon},
        name="rsds_deaccum",
        attrs={"units": "W m-2", "method": "diff per cycle / (6*3600) — correct"},
    )
    return rsds


# ── print comparison table ────────────────────────────────────────────────────

def compare(ssrd_raw, rsds_naive, rsds_deaccum, hourly_nc_path):
    log.info("\n" + "=" * 60)
    log.info("DOMAIN-MEAN DAILY MEAN rsds  (W m-2)")
    log.info("=" * 60)

    def daily_mean_wm2(da):
        return float(da.resample(time="1D").mean().mean())

    naive_val   = daily_mean_wm2(rsds_naive)
    deaccum_val = daily_mean_wm2(rsds_deaccum)

    log.info("  Naive   (/6*3600, current) : %8.2f W m-2", naive_val)
    log.info("  Deaccum (correct)          : %8.2f W m-2", deaccum_val)
    log.info("  Ratio naive / deaccum      : %8.3f", naive_val / deaccum_val)

    if hourly_nc_path and Path(hourly_nc_path).exists():
        ds_h = xr.open_dataset(hourly_nc_path)
        hrly_val = float(ds_h["rsds"].resample(time="1D").mean().mean())
        log.info("  Hourly NC (W m-2)          : %8.2f W m-2", hrly_val)
        log.info("  Ratio naive   / hourly     : %8.3f", naive_val   / hrly_val)
        log.info("  Ratio deaccum / hourly     : %8.3f", deaccum_val / hrly_val)
        ds_h.close()
    else:
        log.info("  (no hourly NC provided for comparison)")

    log.info("=" * 60)

    # Also print the per-UTC-hour mean to see the diurnal structure
    log.info("\nDOMAIN-MEAN by UTC hour (W m-2):")
    log.info("  %4s  %10s  %10s", "UTC", "naive", "deaccum")
    for h in [0, 6, 12, 18]:
        sel_n = rsds_naive.sel(  time=rsds_naive.time.dt.hour   == h)
        sel_d = rsds_deaccum.sel(time=rsds_deaccum.time.dt.hour == h)
        log.info("  %4d  %10.2f  %10.2f", h, float(sel_n.mean()), float(sel_d.mean()))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year",       type=int, default=2020,
                   help="Year to download (default: 2020)")
    p.add_argument("--out-dir",    type=Path, default=Path("data/raw/era5_test"),
                   help="Where to save the test GRIB")
    p.add_argument("--hourly-nc",  type=str,  default="data/proc/era5/era5_india_2025_hourly.nc",
                   help="Path to existing hourly NC for comparison")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    default_out = args.out_dir / f"era5_india_{args.year}_6h_ssrd_test.grib"

    # 1. Reuse existing GRIB or download
    grib_path = get_or_download_grib(args.year, default_out)

    # 2. Load raw
    log.info("\nLoading raw GRIB …")
    ssrd_raw = load_raw_ssrd(grib_path)

    # 3. Convert
    log.info("\nApplying conversions …")
    rsds_naive   = convert_naive(ssrd_raw)
    rsds_deaccum = convert_deaccum(ssrd_raw)

    # 4. Compare
    compare(ssrd_raw, rsds_naive, rsds_deaccum, args.hourly_nc)

    log.info("\nNote: GRIB kept at %s for manual inspection.", grib_path)
    log.info("Run: python -c \"import cfgrib, xarray as xr; ds=xr.open_dataset('%s', engine='cfgrib'); print(ds)\"", grib_path)


if __name__ == "__main__":
    main()
