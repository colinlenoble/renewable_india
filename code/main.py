#!/usr/bin/env python3
"""
Renewable India — full pipeline driver.

Runs, in order, with the configuration below:
    bias_correction  pipeline/bias_correction_qdm.py     ERA5-daily QDM bias correction of the GCM
    diurnal_fit      pipeline/downscale_hourly.py fit     K-means diurnal library from ERA5 hourly
    diurnal_apply    pipeline/downscale_hourly.py apply   daily BC GCM -> hourly (historical + SSPs)
    cf               pipeline/compute_cf.py               hourly wCF / sCF + annual state means
    wcf_variants     pipeline/compute_wcf_variants.py     hourly wCF per turbine assumption
                                                          (conservative / optimistic, see CONFIG)
    cf_states        pipeline/aggregate_cf_states.py      hourly state series of wCF / sCF
    tas_states       pipeline/aggregate_tas_states.py     hourly state series of tas

ERA5 -> GCM grid (bias-correction reference and diurnal library): conservative
re-aggregation weighted by operating capacity per ERA5 pixel (sfcWind: wind,
rsds: solar, tas: area only), plus a small capacity jitter so cells without
capacity keep their area-weighted mean.
Wind CF: 10 m wind -> 150 m with the per-pixel ERA5 shear exponent.
State aggregation (wCF and sCF alike): area-weighted, area-weighted over
regions buffered by 50 km, and weighted by operating capacity per pixel
(GEM Global Wind / Solar Power Trackers, + the same capacity jitter so no
state is left without a capacity-weighted value).

The raw-data downloads (pipeline/download_*.py) are not part of this driver:
they need CDS / ESGF network access and are run once, beforehand.

Usage (from anywhere; paths are resolved from the repository root)
-----
python code/main.py                              # all steps
python code/main.py --steps cf cf_states         # a subset, in pipeline order
python code/main.py --steps wcf_variants         # only the turbine-assumption wCF files
python code/main.py --dry-run                    # print the commands only
python code/main.py --aggregate-only             # redo only the state aggregations
                                                 # from the existing CF / hourly files
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]      # repository root
PIPELINE = ROOT / "code" / "pipeline"

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Model / scenarios ─────────────────────────────────────────────────────
    "gcm":         "GFDL-ESM4",
    "run":         "r1i1p1f1",
    "ssps":        ["ssp126", "ssp245", "ssp370", "ssp585"],
    "train_start": "1980-01-01",
    "train_end":   "2010-12-31",

    # ── Inputs ────────────────────────────────────────────────────────────────
    "era5_daily_dir":  ROOT / "data/raw/era5_daily",
    "era5_hourly_glob": str(ROOT / "data/raw/era5/era5_india_*_hourly.nc"),
    "cmip_dir":        ROOT / "data/raw/{gcm}",
    "shapefile":       ROOT / "INDIA_STATES.geojson",
    "region_col":      "STNAME_SH",
    "shear_file":      ROOT / "aux_data/shear_exponent_local_1982-01-01_2001-12-31.nc",
    "wind_tracker":    ROOT / "aux_data/Global-Wind-Power-Tracker-February-2026.xlsx",
    "solar_tracker":   ROOT / "aux_data/Global-Solar-Power-Tracker-February-2026.xlsx",

    # ── Outputs ───────────────────────────────────────────────────────────────
    "bc_dir":          ROOT / "data/proc/{gcm}",           # bias-corrected daily files
    "library":         ROOT / "data/proc/era5/diurnal_library_{gcm}.nc",
    "hourly_dir":      ROOT / "data/proc/cmip6_hourly",
    "cf_root":         ROOT / "data/proc/cf",               # CF files in <cf_root>/<gcm>/
    "cf_states_dir":   ROOT / "data/results/cf_states",
    "tas_states_dir":  ROOT / "data/results/tas_states",

    # ── Method parameters ─────────────────────────────────────────────────────
    "nquantiles":      25,        # QDM
    "n_clusters":      30,        # diurnal library
    "doy_window":      30,
    "seed":            42,
    "hub_height":      150,       # m
    "wind_method":     "shear_local",
    # wcf_variants: name -> hub height (m), rated / cut-in / cut-out speed (m s-1)
    "wind_variants": {
        "conservative": {"hub_height": 120, "vr": 13.0, "vci": 3.5, "vco": 25.0},
        "optimistic":   {"hub_height": 150, "vr": 11.0, "vci": 3.0, "vco": 27.0},
    },
    "buffer_km":       50,
    "capacity_jitter_mw": 1e-3,   # capacity floor (MW per pixel) in all capacity weightings
    "skip_validation": False,     # daily BC validation plots in compute_cf
    "overwrite":       True,      # recompute outputs that already exist (else skip them)
}

STEPS = ["bias_correction", "diurnal_fit", "diurnal_apply", "cf", "wcf_variants",
         "cf_states", "tas_states"]
AGGREGATION_STEPS = ["cf", "cf_states", "tas_states"]   # run with --aggregate-only

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


def _p(key: str, c: dict) -> Path:
    """Config path with {gcm} filled in."""
    return Path(str(c[key]).format(gcm=c["gcm"]))


def build_commands(c: dict, aggregate_only: bool = False
                   ) -> dict[str, tuple[list[str], list[Path]]]:
    """{step: (argv, input files that must exist before the step runs)}"""
    py, gcm = sys.executable, c["gcm"]
    scenarios = ["historical", *c["ssps"]]
    cf_dir = _p("cf_root", c) / gcm
    force = ["--force"] if c["overwrite"] else []
    trackers = [_p("wind_tracker", c), _p("solar_tracker", c)]
    capacity = ["--wind-capacity-file", str(trackers[0]),
                "--solar-capacity-file", str(trackers[1])]
    jitter = ["--capacity-jitter", str(c["capacity_jitter_mw"])]

    # compute_cf.py: aggregation-only reads the existing CF files, not the hourly inputs
    cf_inputs = [_p("shapefile", c), _p("wind_tracker", c), _p("solar_tracker", c)]
    if aggregate_only:
        cf_inputs += [cf_dir / f"{v}_{gcm}_historical_hourly.nc" for v in ("wCF", "sCF")]
    else:
        cf_inputs += [_p("hourly_dir", c) / f"{gcm}_historical_hourly.nc", _p("shear_file", c)]

    return {
        "bias_correction": ([
            py, str(PIPELINE / "bias_correction_qdm.py"),
            "--gcm", gcm, "--run", c["run"],
            "--era5-dir", str(_p("era5_daily_dir", c)),
            "--cmip-dir", str(_p("cmip_dir", c)),
            "--out-dir", str(_p("bc_dir", c)),
            "--ssps", *c["ssps"],
            "--train-start", c["train_start"], "--train-end", c["train_end"],
            "--nquantiles", str(c["nquantiles"]),
            *capacity, *jitter,
            *force,
        ], [_p("era5_daily_dir", c), _p("cmip_dir", c), *trackers]),

        "diurnal_fit": ([
            py, str(PIPELINE / "downscale_hourly.py"), "fit",
            "--era5-nc", c["era5_hourly_glob"],
            "--gcm-grid", str(_p("bc_dir", c) / f"tas_{gcm}_historical_bc.nc"),
            "--out-library", str(_p("library", c)),
            "--n-clusters", str(c["n_clusters"]),
            *capacity, *jitter,
        ], [_p("bc_dir", c) / f"tas_{gcm}_historical_bc.nc", *trackers]),

        "diurnal_apply": ([
            py, str(PIPELINE / "downscale_hourly.py"), "apply",
            "--library", str(_p("library", c)),
            "--bc-dir", str(_p("bc_dir", c)),
            "--gcm", gcm, "--run", c["run"],
            "--ssps", *scenarios,
            "--out-dir", str(_p("hourly_dir", c)),
            "--doy-window", str(c["doy_window"]),
            "--seed", str(c["seed"]),
            *force,
        ], [_p("library", c)]),

        "cf": ([
            py, str(PIPELINE / "compute_cf.py"),
            "--bc-dir", str(_p("bc_dir", c)),
            "--hourly-dir", str(_p("hourly_dir", c)),
            "--cmip-dir", str(_p("cmip_dir", c)),
            "--shapefile", str(_p("shapefile", c)),
            "--out-dir", str(cf_dir),
            "--gcm", gcm, "--run", c["run"],
            "--ssps", *c["ssps"],
            "--train-start", c["train_start"], "--train-end", c["train_end"],
            "--region-col", c["region_col"],
            "--hub-height", str(c["hub_height"]),
            "--wind-method", c["wind_method"],
            "--shear-file", str(_p("shear_file", c)),
            "--buffer-km", str(c["buffer_km"]),
            *capacity, *jitter,
            *(["--skip-validation"] if c["skip_validation"] else []),
            *(["--aggregate-only"] if aggregate_only else force),
        ], cf_inputs),

        "wcf_variants": ([
            py, str(PIPELINE / "compute_wcf_variants.py"),
            "--hourly-dir", str(_p("hourly_dir", c)),
            "--out-dir", str(cf_dir),
            "--gcm", gcm,
            "--scenarios", *scenarios,
            "--wind-method", c["wind_method"],
            "--shear-file", str(_p("shear_file", c)),
            *[a for name, v in c["wind_variants"].items()
              for a in ("--variant", name, *(str(v[k]) for k in ("hub_height", "vr", "vci", "vco")))],
            *force,
        ], [_p("hourly_dir", c) / f"{gcm}_{s}_hourly.nc" for s in scenarios]
           + [_p("shear_file", c)]),

        "cf_states": ([
            py, str(PIPELINE / "aggregate_cf_states.py"),
            "--cf-dir", str(_p("cf_root", c)),
            "--shapefile", str(_p("shapefile", c)),
            "--out-dir", str(_p("cf_states_dir", c)),
            "--gcm", gcm,
            "--scenarios", *scenarios,
            "--variables", "wCF", "sCF",
            "--region-col", c["region_col"],
            "--buffer-km", str(c["buffer_km"]),
            *capacity, *jitter,
        ], [cf_dir / f"wCF_{gcm}_historical_hourly.nc", *trackers]),

        "tas_states": ([
            py, str(PIPELINE / "aggregate_tas_states.py"),
            "--hourly-dir", str(_p("hourly_dir", c)),
            "--shapefile", str(_p("shapefile", c)),
            "--out-dir", str(_p("tas_states_dir", c)),
            "--gcm", gcm,
            "--scenarios", *scenarios,
            "--region-col", c["region_col"],
        ], [_p("hourly_dir", c) / f"{gcm}_historical_hourly.nc"]),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Renewable India pipeline driver")
    p.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS,
                   help="Steps to run (always executed in pipeline order)")
    p.add_argument("--gcm", default=None, help=f"Override GCM (default {CONFIG['gcm']})")
    p.add_argument("--run", default=None, help=f"Override member (default {CONFIG['run']})")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running")
    p.add_argument("--aggregate-only", action="store_true",
                   help=f"Only redo the state aggregations ({', '.join(AGGREGATION_STEPS)}) "
                        "from existing CF / hourly files — no bias correction, "
                        "downscaling or CF computation")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None,
                   help=f"Recompute outputs that already exist; --no-overwrite skips them "
                        f"(default {CONFIG['overwrite']})")
    return p.parse_args()


def main():
    args = parse_args()
    c = dict(CONFIG)
    if args.gcm:
        c["gcm"] = args.gcm
    if args.run:
        c["run"] = args.run
    if args.overwrite is not None:
        c["overwrite"] = args.overwrite

    commands = build_commands(c, args.aggregate_only)
    allowed = AGGREGATION_STEPS if args.aggregate_only else STEPS
    for step in [s for s in allowed if s in args.steps]:
        argv, inputs = commands[step]
        log.info("══ %s ══", step)
        log.info("  %s", " ".join(argv))
        if args.dry_run:
            continue
        missing = [str(f) for f in inputs if not Path(f).exists()]
        if missing:
            log.error("  Missing input(s) for %s:\n    %s", step, "\n    ".join(missing))
            sys.exit(1)
        result = subprocess.run(argv, cwd=ROOT)
        if result.returncode != 0:
            log.error("  %s failed (exit code %d) — stopping", step, result.returncode)
            sys.exit(result.returncode)
    log.info("Done.")


if __name__ == "__main__":
    main()
