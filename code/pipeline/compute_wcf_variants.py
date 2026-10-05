#!/usr/bin/env python3
"""
Hourly wind capacity factor (wCF) for several turbine assumptions, from the
hourly downscaled 10 m wind (downscale_hourly.py apply output).

Each --variant gives a name, hub height and power-curve speeds; the 10 m wind
is extrapolated to that hub height with the same shear method as compute_cf.py
(per-pixel ERA5 alpha by default) and passed through the same piecewise power
curve (wind_cf_hourly). No solar CF, validation or state aggregation here.

Input:  <hourly-dir>/<gcm>_<scenario>_hourly.nc                    (sfcWind)
Output: <out-dir>/wCF_<variant>_<gcm>_<scenario>_hourly.nc

Usage
-----
python compute_wcf_variants.py \\
    --hourly-dir data/proc/cmip6_hourly --out-dir data/proc/cf/CanESM5 \\
    --gcm CanESM5 --scenarios historical ssp245 ssp585 \\
    --shear-file aux_data/shear_exponent_local_1982-01-01_2001-12-31.nc \\
    --variant conservative 120 13 3.5 25 \\
    --variant optimistic   150 11 3   27
"""

import argparse
import logging
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_cf import (  # noqa: E402
    CFConfig, DEFAULT_SHEAR_FILE, ENCODING_HOURLY, WIND_METHODS,
    load_hourly, load_local_shear_exponent, wind_cf_hourly,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_variant(v: list[str]) -> tuple[str, CFConfig]:
    name, hub, vr, vci, vco = v
    return name, CFConfig(hub_height=float(hub), vr=float(vr),
                          vci=float(vci), vco=float(vco))


def parse_args():
    p = argparse.ArgumentParser(description="Hourly wCF for several turbine assumptions")
    p.add_argument("--hourly-dir", required=True, type=Path,
                   help="Directory with <gcm>_<scenario>_hourly.nc (sfcWind)")
    p.add_argument("--out-dir",    required=True, type=Path)
    p.add_argument("--gcm",        default="CanESM5")
    p.add_argument("--scenarios",  nargs="+", default=["historical", "ssp245", "ssp585"])
    p.add_argument("--variant",    nargs=5, action="append", required=True,
                   metavar=("NAME", "HUB_M", "VR", "VCI", "VCO"),
                   help="Turbine assumption: name, hub height (m), rated, "
                        "cut-in and cut-out wind speeds (m s-1); repeatable")
    p.add_argument("--wind-method", choices=WIND_METHODS, default="shear_local")
    p.add_argument("--shear-file",  type=Path, default=DEFAULT_SHEAR_FILE)
    p.add_argument("--force", action="store_true",
                   help="Overwrite wCF files that already exist")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for v in args.variant:
        name, cfg = parse_variant(v)
        cfg.wind_method, cfg.shear_file = args.wind_method, args.shear_file
        variants.append((name, cfg))
        log.info("Variant %-12s hub %.0f m, vci %.1f, vr %.1f, vco %.1f m s-1",
                 name, cfg.hub_height, cfg.vci, cfg.vr, cfg.vco)

    alpha = None
    for scenario in args.scenarios:
        log.info("━━━ %s ━━━", scenario)
        try:
            ds = load_hourly(args.hourly_dir, args.gcm, scenario)
        except FileNotFoundError as exc:
            log.error("  %s — skipping", exc)
            continue
        if alpha is None and args.wind_method == "shear_local":
            log.info("Loading local shear exponent from %s …", args.shear_file)
            alpha = load_local_shear_exponent(ds.isel(time=0), variants[0][1])
        enc_chunks = (24, ds.sizes["lat"], ds.sizes["lon"])
        for name, cfg in variants:
            out_nc = args.out_dir / f"wCF_{name}_{args.gcm}_{scenario}_hourly.nc"
            if out_nc.exists() and not args.force:
                log.info("  %s already exists — skipping", out_nc.name)
                continue
            wcf = wind_cf_hourly(ds["sfcWind"], cfg, alpha)
            wcf.attrs.update(variant=name, vr_ms=cfg.vr, vci_ms=cfg.vci, vco_ms=cfg.vco)
            log.info("  Writing %s …", out_nc.name)
            wcf.to_netcdf(out_nc, encoding={"wCF": {**ENCODING_HOURLY, "chunksizes": enc_chunks}})
            log.info("  → %s", out_nc.name)
        ds.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
