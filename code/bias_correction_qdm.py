#!/usr/bin/env python3
"""
QDM bias correction — ERA5 reference → GCM historical → future SSPs.

Pipeline (one variable at a time to keep RAM low):
  1. Load ERA5, drop extra coords, convert to noleap calendar
  2. Load GCM historical (training period)
  3. Regrid ERA5 (0.25°) → GCM grid with xesmf bilinear
  4. Align time axes on common dates
  5. Train QuantileDeltaMapping (xclim)
  6. Load each SSP future, apply QDM, write NetCDF

Usage
-----
python bias_correction_qdm.py \\
    --gcm        CanESM5          \\
    --run        r10i1p1f1        \\
    --era5-dir   /data/raw/era5_daily    \\
    --cmip-dir   /data/raw/CanESM5      \\
    --out-dir    /data/proc/cmip6_bc    \\
    --ssps       ssp245 ssp585          \\
    --train-start 1980-01-01            \\
    --train-end   2010-12-31            \\
    --nquantiles  50
"""

import os
import sys
import glob
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.calendar_ops import convert_calendar
from xclim import sdba
from xclim.sdba import processing as sdba_proc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Variable configuration ────────────────────────────────────────────────────────
# key → (kind, use_log, jitter_lower, units)
VAR_CFG = {
    "tas":     ("+", False, None,  "K"),
    "tasmax":  ("+", False, None,  "K"),
    "sfcWind": ("+", True,  1e-6,  "m s-1"),
    "rsds":    ("+", True,  1e-6,  "W m-2"),
}


# ── I/O helpers ───────────────────────────────────────────────────────────────────

def load_era5(era5_dir: Path, vname: str, units: str, train: slice) -> xr.Dataset:
    """Load ERA5, strip GRIB scalar coords, convert to noleap, drop leap-day NaN."""
    path = era5_dir / f"era5_{vname}_1980_2020.nc"
    ds = (
        xr.open_dataset(path)
        .sel(time=train)
        .sortby("lat").sortby("lon")
    )
    ds = ds.drop_vars(
        [c for c in ds.coords if c not in {"time", "lat", "lon"}],
        errors="ignore",
    )
    ds = convert_calendar(ds, "noleap", align_on="date")
    ds = ds.dropna(dim="time", how="all")
    ds[vname].attrs["units"] = units
    return ds


def load_cmip(cmip_dir: Path, var: str, gcm: str, scenario: str,
              run: str, units: str, time_slice: slice = None) -> xr.DataArray:
    """Load one CMIP6 variable, clean coordinates, optionally slice time."""
    pattern = str(cmip_dir / f"{var}_day_{gcm}_{scenario}_{run}*_india.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    da = xr.open_mfdataset(files, combine="by_coords")[var]
    da = da.drop_vars(
        [c for c in da.coords
         if c not in {"time", "lat", "lon", "latitude", "longitude"}],
        errors="ignore",
    )
    da["time"] = da.indexes["time"].floor("D")
    da.attrs["units"] = units
    da = da.sortby("lat").sortby("lon").sortby("time")
    if time_slice is not None:
        da = da.sel(time=time_slice)
    return da


def load_wind_cmip(cmip_dir: Path, gcm: str, scenario: str,
                   run: str, time_slice: slice = None) -> xr.DataArray:
    """Compute sfcWind = hypot(uas, vas) from CMIP6 files."""
    uas = load_cmip(cmip_dir, "uas", gcm, scenario, run, "m s-1", time_slice)
    vas = load_cmip(cmip_dir, "vas", gcm, scenario, run, "m s-1", time_slice)
    ws = np.hypot(uas, vas).rename("sfcWind")
    ws.attrs["units"] = "m s-1"
    del uas, vas
    return ws


# ── Date-alignment helper ─────────────────────────────────────────────────────────

def align_common_dates(da_a: xr.DataArray, da_b: xr.DataArray):
    """
    Return (da_a_aligned, da_b_aligned) sharing only common YYYY-MM-DD dates.
    Works with both cftime and numpy datetime64.
    """
    dates_a = pd.Index([str(t)[:10] for t in da_a.time.values])
    dates_b = pd.Index([str(t)[:10] for t in da_b.time.values])
    common = dates_a.intersection(dates_b)
    idx_a = [i for i, t in enumerate(da_a.time.values) if str(t)[:10] in common]
    idx_b = [i for i, t in enumerate(da_b.time.values) if str(t)[:10] in common]
    da_a = da_a.isel(time=idx_a)
    da_b = da_b.isel(time=idx_b)
    # assign da_b's time coords to da_a so xclim sees a single calendar
    da_a = da_a.assign_coords(time=da_b.time.values)
    return da_a, da_b, len(common)


# ── Log-space transform ───────────────────────────────────────────────────────────

def jitter_log(da: xr.DataArray, lower: float, unit: str) -> xr.DataArray:
    da = sdba_proc.jitter(da, lower=f"{lower} {unit}", minimum=f"0 {unit}")
    return sdba_proc.to_additive_space(da, lower_bound=f"0 {unit}", trans="log")


# ── Main ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="QDM bias correction (ERA5 reference, CMIP6 target)"
    )
    p.add_argument("--gcm",          default="CanESM5",
                   help="GCM name matching CMIP6 file naming (default: CanESM5)")
    p.add_argument("--run",          default="r10i1p1f1",
                   help="Ensemble member (default: r10i1p1f1)")
    p.add_argument("--era5-dir",     required=True, type=Path,
                   help="Directory containing era5_<var>_1980_2020.nc files")
    p.add_argument("--cmip-dir",     required=True, type=Path,
                   help="Directory containing CMIP6 NetCDF files")
    p.add_argument("--out-dir",      required=True, type=Path,
                   help="Output directory for bias-corrected files")
    p.add_argument("--ssps",         nargs="+", default=["ssp245", "ssp585"],
                   help="SSP scenarios to process (default: ssp245 ssp585)")
    p.add_argument("--train-start",  default="1980-01-01",
                   help="Training period start (default: 1980-01-01)")
    p.add_argument("--train-end",    default="2010-12-31",
                   help="Training period end (default: 2010-12-31)")
    p.add_argument("--nquantiles",   type=int, default=50,
                   help="Number of quantiles for QDM (default: 50)")
    p.add_argument("--env-dir",      type=Path, default=None,
                   help="Conda env root for ESMFMKFILE (default: sys.prefix)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── ESMFMKFILE must be set before importing xesmf ─────────────────────────────
    env_dir = Path(args.env_dir) if args.env_dir else Path(sys.prefix)
    mk = env_dir / "Library" / "lib" / "esmf.mk"   # Windows conda
    if not mk.exists():
        mk = env_dir / "lib" / "esmf.mk"            # Linux / Mac
    os.environ["ESMFMKFILE"] = str(mk)
    log.info("ESMFMKFILE = %s  (exists: %s)", mk, mk.exists())

    import xesmf as xe

    args.out_dir.mkdir(parents=True, exist_ok=True)
    TRAIN = slice(args.train_start, args.train_end)

    # ─────────────────────────────────────────────────────────────────────────────
    # Variable loop — load, regrid, train, apply, free
    # ─────────────────────────────────────────────────────────────────────────────
    for vname, (kind, use_log, jitter_low, unit) in VAR_CFG.items():
        log.info("━━━ %s ━━━", vname)

        # 1. ERA5 reference (training period)
        log.info("  Loading ERA5 %s …", vname)
        era5_ds = load_era5(args.era5_dir, vname, unit, TRAIN)
        log.info("  ERA5: %s", dict(era5_ds[vname].sizes))

        # 2. GCM historical (training period)
        log.info("  Loading %s historical %s …", args.gcm, vname)
        if vname == "sfcWind":
            hist_da = load_wind_cmip(args.cmip_dir, args.gcm, "historical",
                                     args.run, TRAIN)
        else:
            hist_da = load_cmip(args.cmip_dir, vname, args.gcm, "historical",
                                args.run, unit, TRAIN)
        log.info("  %s hist: %s", args.gcm, dict(hist_da.sizes))

        # store the reference lat/lon (used later to override future coords)
        ref_lat = hist_da.lat
        ref_lon = hist_da.lon

        # 3. Regrid ERA5 → GCM grid with xesmf
        log.info("  Regridding ERA5 → %s grid …", args.gcm)
        target_grid = (
            hist_da.isel(time=0)
            .drop_vars("time", errors="ignore")
            .to_dataset(name=vname)
        )
        regridder = xe.Regridder(
            era5_ds, target_grid,
            method="bilinear",
            extrap_method="nearest_s2d",
            reuse_weights=False,
        )
        era5_rg = regridder(era5_ds)[vname]
        era5_rg.attrs["units"] = unit
        # override lat/lon to exact GCM float values (avoids tiny FP mismatches)
        era5_rg = era5_rg.assign_coords(lat=ref_lat, lon=ref_lon)
        del era5_ds, target_grid, regridder
        log.info("  ERA5 regridded: %s", dict(era5_rg.sizes))

        # 4. Align on common dates
        era5_rg, hist_da, n_common = align_common_dates(era5_rg, hist_da)
        era5_rg = era5_rg.load()
        hist_da = hist_da.load()
        log.info("  Common training days: %d", n_common)

        # 5. Optional log transform
        if use_log:
            ref_v  = jitter_log(era5_rg, jitter_low, unit)
            hist_v = jitter_log(hist_da, jitter_low, unit)
        else:
            ref_v, hist_v = era5_rg, hist_da

        # 6. Train QDM
        log.info("  Training QDM …")
        QM = sdba.QuantileDeltaMapping.train(
            ref_v, hist_v,
            nquantiles=args.nquantiles,
            kind=kind,
            group="time.month",
        )
        del ref_v, hist_v, era5_rg, hist_da
        log.info("  QDM trained")

        # 7. Apply to each SSP scenario
        for ssp in args.ssps:
            out_path = args.out_dir / f"{vname}_{args.gcm}_{ssp}_bc.nc"
            if out_path.exists():
                log.info("  %s/%s: already exists — skipping", ssp, vname)
                continue

            log.info("  Loading %s %s future …", ssp, vname)
            if vname == "sfcWind":
                fut_da = load_wind_cmip(args.cmip_dir, args.gcm, ssp, args.run)
            else:
                fut_da = load_cmip(args.cmip_dir, vname, args.gcm, ssp,
                                   args.run, unit)
            # override coords to match training grid exactly
            fut_da = fut_da.assign_coords(lat=ref_lat, lon=ref_lon)
            fut_da = fut_da.load()
            log.info("  Future loaded: %s", dict(fut_da.sizes))

            if use_log:
                fut_da = jitter_log(fut_da, jitter_low, unit)

            log.info("  Applying QDM …")
            bc = QM.adjust(fut_da, interp="linear", extrapolation="constant")
            del fut_da

            if use_log:
                bc = sdba_proc.from_additive_space(bc)
                bc.values[bc.values < 1e-5] = 0.0

            ds_out = bc.to_dataset(name=vname)
            ds_out[vname].attrs["units"] = unit
            ds_out.attrs = {
                "description": (f"QDM bias-corrected {vname} — "
                                f"{args.gcm} {args.run} {ssp}"),
                "method": "Quantile Delta Mapping (Cannon et al. 2015) via xclim.sdba",
                "reference": f"ERA5 daily {args.train_start} – {args.train_end}",
                "gcm": args.gcm,
                "run": args.run,
                "ssp": ssp,
            }
            ds_out.to_netcdf(out_path)
            del bc, ds_out
            log.info("  → %s", out_path.name)

        del QM
        log.info("  %s complete", vname)

    log.info("All variables done.")


if __name__ == "__main__":
    main()
