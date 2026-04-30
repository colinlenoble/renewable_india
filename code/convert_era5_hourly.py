#!/usr/bin/env python3
"""
Convert ERA5 hourly GRIB files → NetCDF, then delete the source GRIBs.

What the script does per GRIB file
------------------------------------
1. Load t2m, ssrd, u10, v10 from the GRIB with cfgrib.
2. De-accumulate ssrd (J m-2 accumulated since midnight → W m-2 instantaneous):
       ssrd_h = max(diff(ssrd_acc, prepend=0), 0) / 3600
3. Compute sfcWind = hypot(u10, v10).
4. Write a compressed NetCDF4 file containing {tas, rsds, sfcWind},
   chunked (24, n_lat, n_lon) — one chunk = one day for every cell.
5. Delete the original GRIB file (and its .idx sidecar if present).

Output naming
-------------
Input : era5_india_2020_hourly.grib
Output: era5_india_2020_hourly.nc   (same folder by default)

Usage
-----
# Single file:
python convert_era5_hourly.py \\
    data/raw/era5/era5_india_2025_hourly.grib

# Glob (all years):
python convert_era5_hourly.py \\
    data/raw/era5/era5_india_*.grib \\
    --out-dir data/proc/era5

# Dry-run (no deletion, no write):
python convert_era5_hourly.py data/raw/era5/era5_india_2025_hourly.grib --dry-run
"""

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── GRIB → xr.Dataset ────────────────────────────────────────────────────────

def _rename_latlon(ds: xr.Dataset) -> xr.Dataset:
    rn = {}
    if "latitude" in ds.coords:
        rn["latitude"] = "lat"
    if "longitude" in ds.coords:
        rn["longitude"] = "lon"
    return ds.rename(rn) if rn else ds


def _drop_extra_coords(da: xr.DataArray) -> xr.DataArray:
    drop = [c for c in da.coords if c not in {"time", "lat", "lon"}]
    return da.drop_vars(drop, errors="ignore")


def load_grib(grib_path: Path) -> xr.Dataset:
    """
    Read t2m, ssrd, u10, v10 from a single ERA5 hourly GRIB file.
    Returns Dataset with lat/lon coords (lowercase), time in numpy datetime64.
    ssrd is de-accumulated (per forecast cycle) and converted to W m-2.

    ERA5 CDS structure:
      instant fields (t2m, u10, v10): flat (time, lat, lon) — time = valid_time
      accumulated field (ssrd):       (time, step, lat, lon) — valid_time = time + step
                                       accumulation resets at each cycle (00 / 12 UTC)
    """
    import cfgrib

    # ── instant fields ────────────────────────────────────────────────────────
    instant = {}
    for short_name, xr_name in [("2t", "t2m"), ("10u", "u10"), ("10v", "v10")]:
        ds = cfgrib.open_dataset(
            str(grib_path),
            backend_kwargs={"filter_by_keys": {"shortName": short_name}},
            indexpath=None,
        )
        ds = _rename_latlon(ds)
        instant[xr_name] = _drop_extra_coords(ds[xr_name])

    t_ref = pd.DatetimeIndex(instant["t2m"].time.values)
    lat   = instant["t2m"].lat.values
    lon   = instant["t2m"].lon.values

    # ── accumulated ssrd ──────────────────────────────────────────────────────
    ds_acc = cfgrib.open_dataset(
        str(grib_path),
        backend_kwargs={"filter_by_keys": {"shortName": "ssrd"}},
        indexpath=None,
    )
    ds_acc = _rename_latlon(ds_acc)

    ssrd_acc = ds_acc["ssrd"].values          # (n_cycles, n_steps, lat, lon)
    # diff along step axis per cycle; prepend a zero-column so step 1 ≡ step1 - 0
    ssrd_h = np.maximum(
        np.diff(ssrd_acc, axis=1, prepend=np.zeros_like(ssrd_acc[:, :1])), 0
    ) / 3600.0                                # J m-2 → W m-2

    # valid_time for each (cycle, step) cell
    valid_time = ds_acc["valid_time"].values  # (n_cycles, n_steps), datetime64
    vt_flat    = valid_time.flatten()
    ssrd_flat  = ssrd_h.reshape(-1, len(lat), len(lon))

    # sort and select only the times that match the instant variables
    sort_idx  = np.argsort(vt_flat)
    vt_sorted = vt_flat[sort_idx]
    ssrd_sorted = ssrd_flat[sort_idx]

    # align to t_ref using searchsorted
    idx = np.searchsorted(vt_sorted, t_ref.values)
    idx = np.clip(idx, 0, len(vt_sorted) - 1)
    matched = vt_sorted[idx] == t_ref.values
    if not matched.all():
        missing = (~matched).sum()
        raise ValueError(
            f"{missing} ssrd times could not be matched to instant-variable times"
        )

    ssrd_da = xr.DataArray(
        ssrd_sorted[idx].astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={"time": t_ref.values, "lat": lat, "lon": lon},
    )

    # ── assemble output dataset ───────────────────────────────────────────────
    ds_out = xr.Dataset({
        "tas":     _drop_extra_coords(instant["t2m"]),
        "rsds":    ssrd_da,
        "sfcWind": _drop_extra_coords(
            np.hypot(instant["u10"], instant["v10"]).rename("sfcWind")
        ),
    })

    ds_out["tas"].attrs    = {"units": "K",     "long_name": "2 m temperature"}
    ds_out["rsds"].attrs   = {"units": "W m-2", "long_name": "Surface downwelling shortwave radiation"}
    ds_out["sfcWind"].attrs = {"units": "m s-1","long_name": "Near-surface wind speed (10 m)"}

    return ds_out.sortby("lat").sortby("lon")


# ── Write NetCDF ──────────────────────────────────────────────────────────────

def write_nc(ds: xr.Dataset, nc_path: Path) -> None:
    """Write Dataset to a compressed NetCDF4 file (zlib level 4)."""
    n_lat = ds.dims["lat"]
    n_lon = ds.dims["lon"]
    chunk_time = 24   # one day per chunk along time

    encoding = {
        v: {
            "zlib": True,
            "complevel": 4,
            "chunksizes": (chunk_time, n_lat, n_lon),
            "dtype": "float32",
        }
        for v in ds.data_vars
    }
    ds.to_netcdf(nc_path, encoding=encoding)


# ── Main ──────────────────────────────────────────────────────────────────────

def convert_one(grib_path: Path, out_dir: Path, dry_run: bool) -> None:
    nc_path = out_dir / (grib_path.stem + ".nc")

    if nc_path.exists():
        log.info("  %s already exists — skipping", nc_path.name)
        return

    log.info("Loading %s …", grib_path.name)
    ds = load_grib(grib_path)
    log.info(
        "  %d hours  |  lat %d  lon %d  |  vars: %s",
        ds.dims["time"], ds.dims["lat"], ds.dims["lon"],
        list(ds.data_vars),
    )

    if dry_run:
        log.info("  [dry-run] would write → %s", nc_path)
        log.info("  [dry-run] would delete %s", grib_path)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("  Writing → %s …", nc_path.name)
    write_nc(ds, nc_path)
    log.info("  NetCDF written (%.1f MB)", nc_path.stat().st_size / 1e6)

    # Remove GRIB and its cfgrib index sidecar
    grib_path.unlink()
    log.info("  Deleted %s", grib_path.name)
    for candidate in grib_path.parent.glob(grib_path.name + "*.idx"):
        candidate.unlink()
        log.info("  Deleted %s", candidate.name)


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert ERA5 hourly GRIB to NetCDF and delete source GRIB"
    )
    p.add_argument(
        "grib_patterns", nargs="+",
        help="GRIB file paths or glob patterns (e.g. data/raw/era5/*.grib)",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for .nc files (default: same dir as GRIB)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing or deleting anything",
    )
    return p.parse_args()


def main():
    args = parse_args()

    grib_files = []
    for pat in args.grib_patterns:
        matched = sorted(glob.glob(pat))
        if not matched:
            log.warning("No files matched: %s", pat)
        grib_files.extend(matched)

    if not grib_files:
        raise SystemExit("No GRIB files found.")

    log.info("Files to convert: %d", len(grib_files))
    for grib_str in grib_files:
        grib_path = Path(grib_str)
        out_dir = args.out_dir if args.out_dir else grib_path.parent
        convert_one(grib_path, out_dir, dry_run=args.dry_run)

    log.info("Done.")


if __name__ == "__main__":
    main()
