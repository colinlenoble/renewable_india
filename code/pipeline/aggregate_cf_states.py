#!/usr/bin/env python3
"""
Extract wCF/sCF from the hourly gridded CF output (compute_cf.py) and
aggregate to Indian states (area-weighted, via xagg) — same shapefile
conventions as aggregate_tas_states.py.

Input:  <cf-dir>/<gcm>/{wCF,sCF}_<gcm>_<scenario>_hourly.nc   (compute_cf.py output)
Output: <out-dir>/<gcm>/{wCF,sCF}_<gcm>_<scenario>_states_hourly.csv  (native hourly)
        + ..._states_buffer{km}km_hourly.csv  (if --buffer-km > 0)
        + ..._states_capacity_hourly.csv      (if --wind/solar-capacity-file)
        — same three aggregations as the annual means of compute_cf.py.

Loads data eagerly via h5netcdf + numpy (not dask/netCDF4) — both have shown
DLL/crash issues in this environment; every other script in this session
uses the same pattern successfully.

Usage
-----
python aggregate_cf_states.py \\
    --cf-dir      /gpfs/workdir/shared/juicce/RE_Colin/India/renewable_india/data/proc \\
    --shapefile   /data/INDIA_STATES.geojson \\
    --out-dir     /data/results/cf_states \\
    --gcm         GFDL-ESM4 \\
    --scenarios   historical ssp126 ssp370 \\
    --variables   wCF sCF \\
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

# Same buffer / capacity-weighting logic as the annual means in compute_cf.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_cf import buffer_regions, build_capacity_weights, capacity_weighted_mean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def cf_nc_path(cf_dir: Path, gcm: str, var: str, scenario: str) -> Path:
    """Convention: <cf-dir>/<gcm>/{var}_{gcm}_{scenario}_hourly.nc"""
    return cf_dir / gcm / f"{var}_{gcm}_{scenario}_hourly.nc"


def load_hourly_var(path: Path, var: str) -> xr.Dataset:
    """Hourly CF variable, loaded eagerly, as a small in-memory Dataset (for xagg)."""
    ds = xr.open_dataset(path, engine="h5netcdf")
    out = xr.Dataset(
        {var: (["time", "lat", "lon"], ds[var].values.astype(np.float32))},
        coords={"time": ds["time"].values, "lat": ds["lat"].values, "lon": ds["lon"].values},
    )
    ds.close()
    return out


def _xagg_to_df(ds_hourly, var, wm, gdf, region_col) -> pd.DataFrame:
    ds_out = xa.aggregate(ds_hourly, wm).to_dataset()          # (poly_idx, time)
    return pd.DataFrame(
        ds_out[var].values.T,
        index=pd.DatetimeIndex(ds_out["time"].values),
        columns=gdf[region_col].tolist(),
    )


def aggregate_scenario_var(var, scenario, cf_dir, out_dir, gcm, gdf, wm, region_col,
                           wm_buf=None, buffer_km=0.0, cap_weights=None):
    """
    Hourly state series of *var*, aggregated the same ways as the annual
    means of compute_cf.py: area-weighted (always), area-weighted over the
    buffered polygons (if wm_buf), capacity-weighted (if cap_weights).
    """
    path = cf_nc_path(cf_dir, gcm, var, scenario)
    if not path.exists():
        log.warning("  %s not found — skipping", path)
        return

    log.info("  Loading %s …", path.name)
    ds_hourly = load_hourly_var(path, var)
    log.info("  Aggregating to regions (%d hourly steps) …", ds_hourly.sizes["time"])

    dfs = {"": _xagg_to_df(ds_hourly, var, wm, gdf, region_col)}
    if wm_buf is not None:
        dfs[f"_buffer{buffer_km:g}km"] = _xagg_to_df(ds_hourly, var, wm_buf, gdf, region_col)
    if cap_weights is not None:
        out = capacity_weighted_mean(ds_hourly[var], cap_weights)
        dfs["_capacity"] = pd.DataFrame(
            out.values, index=pd.DatetimeIndex(out["time"].values),
            columns=cap_weights["region"].values,
        )

    gcm_out_dir = out_dir / gcm
    gcm_out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, df in dfs.items():
        out_csv = gcm_out_dir / f"{var}_{gcm}_{scenario}_states{suffix}_hourly.csv"
        df.to_csv(out_csv, index_label="time")
        log.info("  -> %s", out_csv)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract wCF/sCF from hourly gridded CF output and aggregate to Indian states"
    )
    p.add_argument("--cf-dir",     required=True, type=Path,
                    help="Directory with <gcm>/{wCF,sCF}_<gcm>_<scenario>_hourly.nc (compute_cf.py output)")
    p.add_argument("--shapefile",  required=True, type=Path,
                    help="GeoJSON / shapefile of Indian states")
    p.add_argument("--out-dir",    required=True, type=Path,
                    help="Output root; a <gcm> subfolder is created inside it")
    p.add_argument("--gcm",        required=True)
    p.add_argument("--scenarios",  nargs="+", default=["historical", "ssp126", "ssp370"])
    p.add_argument("--variables",  nargs="+", default=["wCF", "sCF"])
    p.add_argument("--region-col", default="STNAME_SH")
    p.add_argument("--buffer-km",  type=float, default=0.0,
                    help="Also aggregate over regions buffered by this distance (km); 0 disables")
    p.add_argument("--wind-capacity-file",  type=Path, default=None,
                    help="GEM Global Wind Power Tracker .xlsx → capacity-weighted wCF")
    p.add_argument("--solar-capacity-file", type=Path, default=None,
                    help="GEM Global Solar Power Tracker .xlsx → capacity-weighted sCF")
    return p.parse_args()


def main():
    args = parse_args()
    gdf = gpd.read_file(args.shapefile).to_crs("EPSG:4326")
    log.info("Shapefile: %d regions", len(gdf))

    # Build the xagg weightmap once, from whichever variable/scenario file is found first
    # (wCF and sCF share the same grid, so one weightmap covers both)
    mask_source = None
    mask_var = None
    for var in args.variables:
        for scn in args.scenarios:
            p = cf_nc_path(args.cf_dir, args.gcm, var, scn)
            if p.exists():
                mask_source, mask_var = p, var
                break
        if mask_source is not None:
            break
    if mask_source is None:
        raise FileNotFoundError(
            f"No CF file found for {args.gcm} in {args.cf_dir} "
            f"(looked for {args.variables} x {args.scenarios})"
        )

    with xr.open_dataset(mask_source, engine="h5netcdf") as _ds:
        ds_grid = _ds.isel(time=0).drop_vars("time", errors="ignore")[[mask_var]]
    log.info("Building xagg weightmap …")
    wm = xa.pixel_overlaps(ds_grid, gdf)
    wm_buf = None
    if args.buffer_km > 0:
        log.info("Building xagg weightmap for regions buffered by %g km …", args.buffer_km)
        wm_buf = xa.pixel_overlaps(ds_grid, buffer_regions(gdf, args.buffer_km))
    cap_weights = {}
    for var, tracker in [("wCF", args.wind_capacity_file), ("sCF", args.solar_capacity_file)]:
        if tracker is not None and var in args.variables:
            log.info("Building %s capacity weights from %s …", var, tracker)
            cap_weights[var], _ = build_capacity_weights(tracker, ds_grid, gdf, args.region_col)

    for var in args.variables:
        for scn in args.scenarios:
            log.info("=== %s / %s ===", var, scn)
            aggregate_scenario_var(var, scn, args.cf_dir, args.out_dir, args.gcm, gdf, wm,
                                   args.region_col, wm_buf, args.buffer_km,
                                   cap_weights.get(var))

    log.info("Done.")


if __name__ == "__main__":
    main()
