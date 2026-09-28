#!/usr/bin/env python3
"""
Extract tas from the hourly downscaled CMIP6 output and aggregate it to
Indian states (area-weighted, via xagg) — same shapefile/region conventions
as compute_cf.py.

Input:  <hourly-dir>/<gcm>_<scenario>_hourly.nc   (produced by downscale_hourly.py)
Output: <out-dir>/<gcm>/tas_<gcm>_<scenario>_states_hourly.csv  (native hourly timestep)

Loads data eagerly via h5netcdf + numpy (not dask/netCDF4) — both have shown
DLL/crash issues in this environment; every other script in this session
uses the same pattern successfully.

Usage
-----
python aggregate_tas_states.py \\
    --hourly-dir  /data/proc/cmip6_hourly \\
    --shapefile   /data/INDIA_STATES.geojson \\
    --out-dir     /data/results/tas_states \\
    --gcm         GFDL-ESM4 \\
    --scenarios   historical ssp126 ssp370 \\
    --region-col  STNAME_SH
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# pyproj needs to find its proj.db. A *different* conda env's activation
# hooks often leave PROJ_DATA/PROJ_LIB pointing at their own (incompatible)
# proj.db in the ambient shell — that mismatch is what raises the CRS error,
# not a missing var, so force it to this interpreter's own copy rather than
# only filling it in when unset (must happen before geopandas/pyproj import).
_proj_dir = Path(sys.prefix) / "Library" / "share" / "proj"
if _proj_dir.exists():
    os.environ["PROJ_DATA"] = str(_proj_dir)
    os.environ["PROJ_LIB"] = str(_proj_dir)

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import xagg as xa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def hourly_nc_path(hourly_dir: Path, gcm: str, scenario: str) -> Path:
    return hourly_dir / f"{gcm}_{scenario}_hourly.nc"


def load_hourly_tas(path: Path) -> xr.Dataset:
    """Hourly tas, loaded eagerly, as a small in-memory Dataset (for xagg)."""
    ds = xr.open_dataset(path, engine="h5netcdf")
    out = xr.Dataset(
        {"tas": (["time", "lat", "lon"], ds["tas"].values.astype(np.float32))},
        coords={"time": ds["time"].values, "lat": ds["lat"].values, "lon": ds["lon"].values},
    )
    ds.close()
    return out


def aggregate_scenario(scenario, hourly_dir, out_dir, gcm, gdf, wm, region_col):
    path = hourly_nc_path(hourly_dir, gcm, scenario)
    if not path.exists():
        log.warning("  %s not found — skipping", path)
        return

    log.info("  Loading %s …", path.name)
    ds_hourly = load_hourly_tas(path)

    log.info("  Aggregating to regions (%d hourly steps) …", ds_hourly.sizes["time"])
    agg = xa.aggregate(ds_hourly, wm)
    ds_out = agg.to_dataset()          # (poly_idx, time)

    df = pd.DataFrame(
        ds_out["tas"].values.T,
        index=pd.DatetimeIndex(ds_out["time"].values),
        columns=gdf[region_col].tolist(),
    )

    #save gridded 
    ds_hourly.to_netcdf(out_dir / f"tas_{gcm}_{scenario}_hourly.nc", engine="h5netcdf")

    gcm_out_dir = out_dir / gcm
    gcm_out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = gcm_out_dir / f"tas_{gcm}_{scenario}_states_hourly.csv"
    df.to_csv(out_csv, index_label="time")
    log.info("  -> %s", out_csv)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract tas from hourly downscaled CMIP6 output and aggregate to Indian states"
    )
    p.add_argument("--hourly-dir", required=True, type=Path,
                    help="Directory with hourly downscaled NetCDF files (downscale_hourly.py output)")
    p.add_argument("--shapefile",  required=True, type=Path,
                    help="GeoJSON / shapefile of Indian states")
    p.add_argument("--out-dir",    required=True, type=Path,
                    help="Output root; a <gcm> subfolder is created inside it")
    p.add_argument("--gcm",        required=True)
    p.add_argument("--scenarios",  nargs="+", default=["historical", "ssp126", "ssp370"])
    p.add_argument("--region-col", default="STNAME_SH")
    return p.parse_args()


def main():
    args = parse_args()
    gdf = gpd.read_file(args.shapefile).to_crs("EPSG:4326")
    log.info("Shapefile: %d regions", len(gdf))

    # Build the xagg weightmap once, from whichever scenario file is found first
    mask_source = next(
        (p for scn in args.scenarios
         if (p := hourly_nc_path(args.hourly_dir, args.gcm, scn)).exists()),
        None,
    )
    if mask_source is None:
        raise FileNotFoundError(f"No hourly file found for {args.gcm} in {args.hourly_dir}")

    with xr.open_dataset(mask_source, engine="h5netcdf") as _ds:
        ds_grid = _ds.isel(time=0).drop_vars("time", errors="ignore")[["tas"]]
    log.info("Building xagg weightmap …")
    wm = xa.pixel_overlaps(ds_grid, gdf)

    for scn in args.scenarios:
        log.info("=== %s ===", scn)
        aggregate_scenario(scn, args.hourly_dir, args.out_dir, args.gcm, gdf, wm, args.region_col)

    log.info("Done.")


if __name__ == "__main__":
    main()
