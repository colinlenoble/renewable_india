"""
Bias-correct CanESM5 CMIP6 daily data against ERA5 daily reference using
Quantile Delta Mapping (QDM, Cannon et al. 2015) from xclim.sdba.

Variables:
  tas     – additive QDM (temperature, Kelvin)
  tasmax  – additive QDM (temperature, Kelvin)
  sfcWind – log-transform then additive QDM (positive-definite)
  rsds    – log-transform then additive QDM (positive-definite)

Training period : 1980–2010 (ERA5 vs CanESM5 historical)
Scenarios       : ssp245, ssp585 (full 2015–2100)
Output          : data/proc/cmip6_bc/{var}_CanESM5_{ssp}_bc.nc

After running this script, use the warming levels CSV in
  aux_data/mathause-cmip_warming_levels-f47853e/
to extract GWL windows from the bias-corrected files.

Run with: conda run -n xr_env python code/bias_correct_cmip6_india.py
"""

import glob
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from xclim import sdba
from xclim.sdba import processing as sdba_proc
from xarray.coding.calendar_ops import convert_calendar

# ─────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────
DATA_RAW   = Path("data/raw")
ERA5_DIR   = DATA_RAW / "era5_daily"
CMIP_DIR   = DATA_RAW / "CanESM5"
OUTPUT_DIR = Path("data/proc/cmip6_bc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GCM        = "CanESM5"
RUN        = "r10i1p1f1"
SSPS       = ["ssp245", "ssp585"]
TRAIN      = slice("1980-01-01", "2010-12-31")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_cmip_var(var: str, scenario: str) -> xr.DataArray:
    """Load one CMIP6 variable, drop auxiliary coords, floor time to date."""
    pattern = str(CMIP_DIR / f"{var}_day_{GCM}_{scenario}_{RUN}*_india.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match: {pattern}")
    da = xr.open_mfdataset(files, combine="by_coords")[var]
    da = da.drop_vars(
        [c for c in da.coords if c not in {"time", "lat", "lon", "latitude", "longitude"}],
        errors="ignore",
    )
    da["time"] = da.indexes["time"].floor("D")
    return da.sortby("lat").sortby("lon").sortby("time")


def jitter_log(da: xr.DataArray, lower: float, unit: str) -> xr.DataArray:
    """Jitter near-zero values and apply log transform for positive-definite vars."""
    da = sdba_proc.jitter(da, lower=f"{lower} {unit}", minimum=f"0 {unit}")
    da = sdba_proc.to_additive_space(da, lower_bound=f"0 {unit}", trans="log")
    return da


def set_units(da: xr.DataArray, unit: str) -> xr.DataArray:
    da.attrs["units"] = unit
    return da

# ─────────────────────────────────────────────
# 1. Load ERA5 daily reference
# ─────────────────────────────────────────────
print("Loading ERA5 daily reference…")

# process_era5_daily.py outputs a single merged file with all variables
# (tas, tasmax, tasmin, uas, vas, rsds) using CF-standard names
ds_era5 = xr.open_dataset(ERA5_DIR / "era5_daily_1980_2020.nc")

era5 = xr.Dataset({
    "tas":     set_units(ds_era5["tas"],                         "K"),
    "tasmax":  set_units(ds_era5["tasmax"],                      "K"),
    "sfcWind": set_units(np.hypot(ds_era5["uas"], ds_era5["vas"]), "m s-1"),
    "rsds":    set_units(ds_era5["rsds"],                        "W m-2"),
})

era5 = era5.sortby("lat").sortby("lon")

# Convert ERA5 standard calendar → noleap (drops Feb-29, matches CMIP6)
era5 = convert_calendar(era5, "noleap", align_on="date")
era5 = era5.sel(time=TRAIN)
print(f"  ERA5 ref: {dict(era5.dims)}, {era5.time.values[0]} → {era5.time.values[-1]}")

# ─────────────────────────────────────────────
# 2. Load CMIP6 historical
# ─────────────────────────────────────────────
print("Loading CanESM5 historical…")

hist_uas = load_cmip_var("uas", "historical")
hist_vas = load_cmip_var("vas", "historical")

hist = xr.Dataset({
    "tas":     set_units(load_cmip_var("tas",    "historical"), "K"),
    "tasmax":  set_units(load_cmip_var("tasmax", "historical"), "K"),
    "sfcWind": set_units(np.hypot(hist_uas, hist_vas),          "m s-1"),
    "rsds":    set_units(load_cmip_var("rsds",   "historical"), "W m-2"),
})
hist = hist.sel(time=TRAIN)
print(f"  CMIP hist: {dict(hist.dims)}, {hist.time.values[0]} → {hist.time.values[-1]}")

# ─────────────────────────────────────────────
# 3. Regrid ERA5 onto CanESM5 grid (bilinear)
# ─────────────────────────────────────────────
print("Regridding ERA5 → CanESM5 grid…")
era5_on_cmip = era5.interp(lat=hist.lat, lon=hist.lon, method="linear")

# Align time indices (noleap strings must match exactly)
era5_times  = pd.DatetimeIndex(era5_on_cmip.time.values).normalize()
hist_times  = pd.DatetimeIndex([pd.Timestamp(str(t)[:10]) for t in hist.time.values])
common_dates = era5_times.intersection(hist_times)

era5_on_cmip = era5_on_cmip.sel(
    time=[t for t in era5_on_cmip.time.values
          if pd.Timestamp(str(t)[:10]) in common_dates]
)
hist = hist.sel(
    time=[t for t in hist.time.values
          if pd.Timestamp(str(t)[:10]) in common_dates]
)
print(f"  Common training days: {len(common_dates)}")

# Assign matching time coordinate so xclim can align them
era5_on_cmip = era5_on_cmip.assign_coords(time=hist.time.values)

# ─────────────────────────────────────────────
# 4. QDM configuration per variable
# ─────────────────────────────────────────────
VAR_CFG = {
    #  var      kind  log    jitter  unit
    "tas":     ("+",  False, None,   "K"),
    "tasmax":  ("+",  False, None,   "K"),
    "sfcWind": ("+",  True,  1e-6,   "m s-1"),
    "rsds":    ("+",  True,  1e-6,   "W m-2"),
}

# ─────────────────────────────────────────────
# 5. Train QDM on historical, apply to each SSP
# ─────────────────────────────────────────────
for ssp in SSPS:
    print(f"\n{'='*55}\nProcessing {ssp}\n{'='*55}")

    # Load future SSP
    fut_uas = load_cmip_var("uas", ssp)
    fut_vas = load_cmip_var("vas", ssp)
    fut = xr.Dataset({
        "tas":     set_units(load_cmip_var("tas",    ssp), "K"),
        "tasmax":  set_units(load_cmip_var("tasmax", ssp), "K"),
        "sfcWind": set_units(np.hypot(fut_uas, fut_vas),   "m s-1"),
        "rsds":    set_units(load_cmip_var("rsds",   ssp), "W m-2"),
    })
    print(f"  Future: {dict(fut.dims)}")

    for vname, (kind, use_log, jitter_low, unit) in VAR_CFG.items():
        out_path = OUTPUT_DIR / f"{vname}_{GCM}_{ssp}_bc.nc"
        if out_path.exists():
            print(f"  {vname}: already exists — skipping")
            continue

        print(f"  {vname}: training QDM (kind='{kind}', log={use_log})…")

        ref_v  = set_units(era5_on_cmip[vname].load(), unit)
        hist_v = set_units(hist[vname].load(),          unit)
        fut_v  = set_units(fut[vname].load(),           unit)

        if use_log:
            ref_v  = jitter_log(ref_v,  jitter_low, unit)
            hist_v = jitter_log(hist_v, jitter_low, unit)
            fut_v  = jitter_log(fut_v,  jitter_low, unit)

        QM = sdba.QuantileDeltaMapping.train(
            ref_v, hist_v,
            nquantiles=50,
            kind=kind,
            group="time.month",
        )

        bc_v = QM.adjust(fut_v, interp="linear", extrapolation="constant")

        if use_log:
            bc_v = sdba_proc.from_additive_space(bc_v)
            bc_v.values[bc_v.values < 1e-5] = 0.0

        ds_out = bc_v.to_dataset(name=vname)
        ds_out[vname].attrs["units"] = unit
        ds_out.attrs = {
            "description":  f"Bias-corrected {vname} — {GCM} {RUN} {ssp}",
            "method":       "Quantile Delta Mapping (Cannon et al. 2015) via xclim.sdba",
            "reference":    "ERA5 daily statistics 1980–2010",
            "training_period": "1980-2010",
            "gcm":          GCM,
            "run":          RUN,
            "ssp":          ssp,
        }
        ds_out.to_netcdf(out_path)
        print(f"    → {out_path}")

print("\nAll bias-corrected files written to", OUTPUT_DIR)

# ─────────────────────────────────────────────
# 6. Extract GWL windows from bias-corrected data
# ─────────────────────────────────────────────
print("\nExtracting Global Warming Level (GWL) windows…")

# Find the warming levels CSV (try both one_ens and all_ens)
wl_dir = Path("aux_data/mathause-cmip_warming_levels-f47853e/warming_levels/cmip6")
wl_csv_files = (
    sorted(glob.glob(str(wl_dir / "csv" / "*1850_1900*.csv"))) or
    sorted(glob.glob(str(wl_dir / "*1850_1900*.csv"))) or
    sorted(glob.glob(str(wl_dir / "**" / "*1850_1900*.csv"), recursive=True))
)
# Prefer all_ens over one_ens for our r10 ensemble
wl_csv_files.sort(key=lambda f: ("all" not in f))
if not wl_csv_files:
    print("  Warming levels CSV not found — skipping GWL extraction")
else:
    wl_csv = wl_csv_files[0]
    print(f"  Using: {wl_csv}")
    wl = pd.read_csv(wl_csv, comment="#", skipinitialspace=True)
    wl.columns = wl.columns.str.strip()

    gwl_dir = OUTPUT_DIR / "gwl"
    gwl_dir.mkdir(exist_ok=True)

    for ssp in SSPS:
        for gwl in [1.5, 2.0, 3.0, 4.0]:
            # Lookup this model/run/scenario/warming_level
            mask = (
                (wl["model"].str.strip() == GCM) &
                (wl["exp"].str.strip() == ssp) &
                (wl["warming_level"].astype(float) == gwl)
            )
            if "ensemble" in wl.columns:
                mask &= wl["ensemble"].str.strip() == RUN

            row = wl[mask]
            if row.empty:
                # Fallback: try any ensemble for this model
                row = wl[
                    (wl["model"].str.strip() == GCM) &
                    (wl["exp"].str.strip() == ssp) &
                    (wl["warming_level"].astype(float) == gwl)
                ]
            if row.empty:
                print(f"  {GCM} {ssp} GWL{gwl}: not found in CSV — skipping")
                continue

            start, end = int(row.iloc[0]["start_year"]), int(row.iloc[0]["end_year"])
            gwl_label = str(gwl).replace(".", "")

            for vname in VAR_CFG:
                bc_file = OUTPUT_DIR / f"{vname}_{GCM}_{ssp}_bc.nc"
                out_gwl  = gwl_dir / f"{vname}_{GCM}_{ssp}_GWL{gwl_label}.nc"
                if not bc_file.exists() or out_gwl.exists():
                    continue
                ds = xr.open_dataset(bc_file)
                ds_gwl = ds.sel(time=slice(str(start), str(end)))
                ds_gwl.attrs["gwl"]        = gwl
                ds_gwl.attrs["gwl_years"]  = f"{start}–{end}"
                ds_gwl.to_netcdf(out_gwl)

            print(f"  {GCM} {ssp} GWL{gwl}: {start}–{end} → {gwl_dir}")

print("Done.")
