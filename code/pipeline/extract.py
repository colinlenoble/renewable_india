#!/usr/bin/env python3
"""
Extract hourly tas from the downscaled files (downscale_hourly.py apply) into
standalone NetCDFs.

Input:  <hourly-dir>/<gcm>_<scenario>_hourly.nc          (rsds, tas, sfcWind)
Output: <hourly-dir>/tas_<gcm>_<run>_<scenario>_hourly.nc (tas only, K)

Usage
-----
python code/pipeline/extract.py --gcm CanESM5 --run r10i1p1f1
python code/pipeline/extract.py --gcm CanESM5 --run r10i1p1f1 --scenarios ssp585
"""

import argparse
import logging
from pathlib import Path

import xarray as xr

ROOT = Path(__file__).resolve().parents[2]      # repository root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def extract(src: Path, out: Path, gcm: str, run: str, scenario: str, var: str = "tas"):
    with xr.open_dataset(src, chunks={"time": 24 * 30}) as ds:
        file_run = ds.attrs.get("run")
        if file_run and file_run != run:
            raise ValueError(f"{src.name} is run {file_run!r}, not {run!r}")
        da = ds[var].astype("float32")
        da.attrs.update({"standard_name": "air_temperature", "units": "K"})
        out_ds = da.to_dataset()
        out_ds.attrs = {
            **ds.attrs,
            "description": f"Hourly {var} — {gcm} {run} {scenario} (extracted from {src.name})",
            "gcm": gcm, "run": run, "ssp": scenario,
        }
        enc = {var: {"zlib": True, "complevel": 4,
                     "chunksizes": (24, da.sizes["lat"], da.sizes["lon"])}}
        tmp = out.with_suffix(".nc.tmp")                 # no half-written file on crash
        out_ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(out)


def parse_args():
    p = argparse.ArgumentParser(description="Extract hourly tas into standalone NetCDFs")
    p.add_argument("--gcm", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--scenarios", nargs="+", default=["historical", "ssp245", "ssp585"])
    p.add_argument("--hourly-dir", type=Path, default=ROOT / "data/proc/cmip6_hourly",
                   help="Folder with <gcm>_<scenario>_hourly.nc; outputs are written there too")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def main():
    args = parse_args()
    for scn in args.scenarios:
        src = args.hourly_dir / f"{args.gcm}_{scn}_hourly.nc"
        out = args.hourly_dir / f"tas_{args.gcm}_{args.run}_{scn}_hourly.nc"
        if not src.exists():
            log.warning("%s not found — skipping", src)
            continue
        if out.exists() and not args.force:
            log.info("%s already exists — skipping (--force to overwrite)", out.name)
            continue
        log.info("%s → %s …", src.name, out.name)
        extract(src, out, args.gcm, args.run, scn)
        log.info("  done")
    log.info("All done.")


if __name__ == "__main__":
    main()
