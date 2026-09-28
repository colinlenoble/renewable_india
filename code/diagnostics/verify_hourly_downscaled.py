#!/usr/bin/env python3
"""
Sanity-check figure for hourly downscaled BC GCM data (output of downscale_hourly.py).

Loads historical / ssp126 / ssp370 hourly files for one GCM, one scenario at a
time (to bound peak memory), and checks:
  - time coverage is complete and strictly hourly
  - value ranges / NaN fraction per variable are physically plausible
  - mean diurnal cycle looks right (rsds ~0 at night, peaks midday; tas peaks
    afternoon; sfcWind flatter)
  - annual-mean series is continuous from historical into the SSPs, with the
    expected divergence between scenarios
  - a short hourly window at one grid cell shows real sub-daily variability
    (not a flat/interpolated curve) — confirms the stochastic profile draw
    actually did something
"""

import gc
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")   # headless: default GUI backend crashes (broken DLL) on this machine
import matplotlib.pyplot as plt

DATA_DIR  = Path(r"E:/temp/GFDL-ESM4")
GCM       = "GFDL-ESM4"
SCENARIOS = ["historical", "ssp126", "ssp370"]
OUT_FIG   = Path("./figures") / f"verify_{GCM}_hourly.png"

VARS      = ["rsds", "tas", "sfcWind"]
VAR_UNITS = {"rsds": "W m$^{-2}$", "tas": "K", "sfcWind": "m s$^{-1}$"}
COLORS    = {"historical": "tab:blue", "ssp126": "tab:green", "ssp370": "tab:red"}


def open_ds(path: Path) -> xr.Dataset:
    # engine pinned to h5netcdf: on this machine, letting xarray try the default
    # netCDF4 backend first leaves the HDF5 DLL state broken for h5netcdf too.
    return xr.open_dataset(path, engine="h5netcdf")


def main():
    # reduced per-scenario results only (small — full datasets are freed after use)
    diurnal = {v: {} for v in VARS}
    annual  = {v: {} for v in VARS}
    window  = {}

    for ssp in SCENARIOS:
        path = DATA_DIR / f"{GCM}_{ssp}_hourly.nc"
        print(f"\nLoading {path.name} …", flush=True)
        ds = open_ds(path)   # stays lazy — only pulled into memory one variable at a time below

        n_hours = ds.sizes["time"]
        steps = np.diff(ds.time.values)
        regular = len(np.unique(steps)) == 1
        print(f"[{ssp}] {ds.time.values[0]} -> {ds.time.values[-1]}  "
              f"({n_hours} steps, multiple of 24: {n_hours % 24 == 0}, "
              f"strictly hourly: {regular})", flush=True)

        if ssp == "historical":
            i_lat, i_lon = ds.sizes["lat"] // 2, ds.sizes["lon"] // 2

        for v in VARS:
            da = ds[v].load()   # one variable at a time (~1.3GB max), freed right after

            vmin, vmax = float(da.min()), float(da.max())
            nan_frac = float(da.isnull().mean())
            print(f"  {v:8s} min={vmin:9.2f}  max={vmax:9.2f}  nan_frac={nan_frac:.4f}", flush=True)

            # reduce spatial dims first (cheap plain mean) — grouping by hour on the
            # full (time, lat, lon) array makes flox allocate a multi-GB intermediate
            da_domain = da.mean(("lat", "lon"))
            diurnal[v][ssp] = da_domain.groupby(da_domain.time.dt.hour).mean()
            annual[v][ssp]  = da_domain.resample(time="YE").mean()

            if ssp == "historical":
                window[v] = da.isel(lat=i_lat, lon=i_lon, time=slice(0, 24 * 5))

            del da
            gc.collect()

        ds.close()
        del ds
        gc.collect()

    print("\nBuilding figure …", flush=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    for row, var in enumerate(VARS):
        # (1) mean diurnal cycle, averaged over space and all days
        ax = axes[row, 0]
        for ssp in SCENARIOS:
            d = diurnal[var][ssp]
            ax.plot(d.hour, d, label=ssp, color=COLORS[ssp])
        ax.set_title(f"{var} — mean diurnal cycle")
        ax.set_xlabel("hour (UTC)")
        ax.set_ylabel(VAR_UNITS[var])
        if row == 0:
            ax.legend()

        # (2) domain-mean annual series, full period (checks scenario continuity/divergence)
        ax = axes[row, 1]
        for ssp in SCENARIOS:
            a = annual[var][ssp]
            ax.plot(a.time, a, color=COLORS[ssp])
        ax.set_title(f"{var} — domain-mean annual series")
        ax.set_ylabel(VAR_UNITS[var])

        # (3) short hourly window at the central grid cell (sub-daily variability check)
        ax = axes[row, 2]
        w = window[var]
        ax.plot(w.time, w, color=COLORS["historical"])
        ax.set_title(f"{var} — 5-day hourly sample (historical)")
        ax.set_ylabel(VAR_UNITS[var])
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"{GCM} hourly downscaled data — sanity check", fontsize=14)
    fig.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print(f"\nSaved figure -> {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
