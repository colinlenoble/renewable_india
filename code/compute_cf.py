#!/usr/bin/env python3
"""
Capacity Factor (CF) computation, regional aggregation, and plots.

Pipeline:
  0. Validation  — alignment check on training period (ERA5 ref vs raw GCM vs BC GCM)
  1. CF grids    — daily wind / solar CF from bias-corrected SSP files
  2. Aggregation — annual mean CF per Indian state
  3. Plots       — time-series per state + 20-year gridded maps

Outputs in <out-dir>:
  validate_{var}_{gcm}.png          per-variable diagnostic (4 panels)
  validate_summary_{gcm}.png        all-variable summary
  {w,s}CF_{gcm}_{ssp}.nc           daily CF grids
  {w,s}CF_{gcm}_{ssp}_states_annual.csv
  timeseries_{w,s}CF_{gcm}.png     annual mean per state, all SSPs
  map_{w,s}CF_{gcm}_{ssp}.png      4-panel 20-year mean maps

Usage
-----
python compute_cf.py \
    --bc-dir    /data/proc/cmip6_bc        \
    --cmip-dir  /data/raw/CanESM5          \
    --shapefile /data/INDIA_STATES.geojson \
    --out-dir   /data/results/cf           \
    --gcm       CanESM5 --run r10i1p1f1   \
    --ssps      ssp245 ssp585             \
    --train-start 1980-01-01              \
    --train-end   2010-12-31              \
    --region-col  STNAME_SH
"""

import argparse
import glob
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────────

@dataclass
class CFConfig:
    vr:         float = 13.0
    vci:        float = 3.5
    vco:        float = 25.0
    hub_height: float = 80.0
    ref_height: float = 10.0
    alpha:      float = 0.143
    gamma:      float = -0.005
    T_ref:      float =  25.0
    G_stc:      float = 1000.0
    c1:         float =   4.3
    c2:         float =   0.943
    c3:         float =   0.028
    c4:         float =  -1.528

CFG = CFConfig()

PERIODS = [(2015, 2034), (2035, 2054), (2055, 2074), (2075, 2094)]
SSP_COLORS = {"ssp245": "steelblue", "ssp585": "firebrick"}
VAR_LABELS = {
    "tas":     "2-m temperature (K)",
    "tasmax":  "Daily max temperature (K)",
    "sfcWind": "10-m wind speed (m s⁻¹)",
    "rsds":    "Surface solar radiation (W m⁻²)",
}


# ── I/O helpers ───────────────────────────────────────────────────────────────────

def load_bc(bc_dir: Path, vname: str, gcm: str, scenario: str) -> xr.DataArray:
    path = bc_dir / f"{vname}_{gcm}_{scenario}_bc.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    return xr.open_dataset(path)[vname]


def load_era5_ref(bc_dir: Path, vname: str) -> xr.DataArray:
    path = bc_dir / f"{vname}_era5_ref.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    return xr.open_dataset(path)[vname]


def load_hist_bc(bc_dir: Path, vname: str, gcm: str) -> xr.DataArray:
    path = bc_dir / f"{vname}_{gcm}_historical_bc.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    return xr.open_dataset(path)[vname]


def load_hist_raw(cmip_dir: Path, vname: str, gcm: str,
                  run: str, train: slice) -> xr.DataArray | None:
    """Load raw GCM historical; returns None if files not found."""
    if vname == "sfcWind":
        return _load_wind_raw(cmip_dir, gcm, run, train)
    pattern = str(cmip_dir / f"{vname}_day_{gcm}_historical_{run}*_india.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        log.warning("  Raw hist not found for %s — skipping raw line", vname)
        return None
    da = xr.open_mfdataset(files, combine="by_coords")[vname]
    da = da.drop_vars(
        [c for c in da.coords if c not in {"time","lat","lon","latitude","longitude"}],
        errors="ignore",
    )
    da["time"] = da.indexes["time"].floor("D")
    return da.sortby("lat").sortby("lon").sortby("time").sel(time=train)


def _load_wind_raw(cmip_dir, gcm, run, train):
    def _one(var):
        pattern = str(cmip_dir / f"{var}_day_{gcm}_historical_{run}*_india.nc")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(pattern)
        da = xr.open_mfdataset(files, combine="by_coords")[var]
        da = da.drop_vars(
            [c for c in da.coords if c not in {"time","lat","lon","latitude","longitude"}],
            errors="ignore",
        )
        da["time"] = da.indexes["time"].floor("D")
        return da.sortby("lat").sortby("lon").sortby("time").sel(time=train)
    try:
        return np.hypot(_one("uas"), _one("vas")).rename("sfcWind")
    except FileNotFoundError as e:
        log.warning("  Raw sfcWind not found (%s) — skipping raw line", e)
        return None


# ── CF formulas ───────────────────────────────────────────────────────────────────

def wind_cf(sfcWind: xr.DataArray, cfg: CFConfig = CFG) -> xr.DataArray:
    w = sfcWind * (cfg.hub_height / cfg.ref_height) ** cfg.alpha
    cf = xr.where(w < cfg.vci,  0.0,
         xr.where(w >= cfg.vco, 0.0,
         xr.where(w >= cfg.vr,  1.0,
                  (w**3 - cfg.vci**3) / (cfg.vr**3 - cfg.vci**3))))
    cf = cf.rename("wCF")
    cf.attrs = {"units": "1", "long_name": "Wind capacity factor",
                "hub_height_m": cfg.hub_height}
    return cf


def solar_cf(tas: xr.DataArray, tasmax: xr.DataArray,
             rsds: xr.DataArray, sfcWind: xr.DataArray,
             cfg: CFConfig = CFG) -> xr.DataArray:
    tas_c    = tas    - 273.15
    tasmax_c = tasmax - 273.15
    T_cell = (cfg.c1
              + cfg.c2 * (tasmax_c + tas_c) / 2.0
              + cfg.c3 * rsds
              + cfg.c4 * sfcWind)
    P_R = 1.0 + cfg.gamma * (T_cell - cfg.T_ref)
    cf = (P_R * rsds / cfg.G_stc).clip(min=0.0)
    cf = cf.rename("sCF")
    cf.attrs = {"units": "1", "long_name": "Solar capacity factor"}
    return cf


# ── Validation diagnostics ────────────────────────────────────────────────────────

def _domain_mean(da: xr.DataArray) -> xr.DataArray:
    return da.mean(["lat", "lon"], skipna=True)

def _monthly_clim(da: xr.DataArray) -> np.ndarray:
    return _domain_mean(da).groupby("time.month").mean().values

def _annual_ts(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    dm = _domain_mean(da).resample(time="YE").mean(skipna=True)
    return dm.time.dt.year.values, dm.values

def _kde(da: xr.DataArray, n_pts: int = 300) -> tuple[np.ndarray, np.ndarray]:
    flat = da.values.ravel()
    flat = flat[~np.isnan(flat)]
    if len(flat) > 500_000:
        flat = np.random.default_rng(0).choice(flat, 500_000, replace=False)
    kde = gaussian_kde(flat, bw_method="scott")
    x = np.linspace(np.percentile(flat, 0.5), np.percentile(flat, 99.5), n_pts)
    return x, kde(x)


def plot_validate_var(vname: str, era5: xr.DataArray, bc: xr.DataArray,
                      raw: xr.DataArray | None, gcm: str, out_path: Path):
    """4-panel validation figure for one variable."""
    months = list("JFMAMJJASOND")
    label  = VAR_LABELS.get(vname, vname)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{label}  |  bias-correction alignment — {gcm}", fontsize=12)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)
    ax_clim     = fig.add_subplot(gs[0, 0])
    ax_pdf      = fig.add_subplot(gs[0, 1])
    ax_ts       = fig.add_subplot(gs[0, 2])
    ax_bias_raw = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_bias_bc  = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax_cb_space = fig.add_subplot(gs[1, 2])
    ax_cb_space.axis("off")

    log.info("    loading data into memory …")
    era5_m = era5.load()
    bc_m   = bc.load()
    raw_m  = raw.load() if raw is not None else None

    # 1. Monthly climatology
    ax_clim.plot(range(1,13), _monthly_clim(era5_m), "k-o",  ms=4, lw=1.5, label="ERA5 ref")
    ax_clim.plot(range(1,13), _monthly_clim(bc_m),   "b-s",  ms=4, lw=1.5, label="GCM bc")
    if raw_m is not None:
        ax_clim.plot(range(1,13), _monthly_clim(raw_m), "r--^", ms=4, lw=1.2, label="GCM raw")
    ax_clim.set_xticks(range(1,13)); ax_clim.set_xticklabels(months, fontsize=8)
    ax_clim.set_ylabel(label, fontsize=8); ax_clim.set_title("Monthly climatology", fontsize=9)
    ax_clim.legend(fontsize=7); ax_clim.grid(True, alpha=0.3)

    # 2. PDF
    xe, ye = _kde(era5_m); xb, yb = _kde(bc_m)
    ax_pdf.plot(xe, ye, "k-",  lw=1.5, label="ERA5 ref")
    ax_pdf.plot(xb, yb, "b-",  lw=1.5, label="GCM bc")
    if raw_m is not None:
        xr_, yr = _kde(raw_m)
        ax_pdf.plot(xr_, yr, "r--", lw=1.2, label="GCM raw")
    ax_pdf.set_xlabel(label, fontsize=8); ax_pdf.set_ylabel("Density", fontsize=8)
    ax_pdf.set_title("PDF of daily values", fontsize=9)
    ax_pdf.legend(fontsize=7); ax_pdf.grid(True, alpha=0.3)

    # 3. Annual time series
    yr_e, ts_e = _annual_ts(era5_m); yr_b, ts_b = _annual_ts(bc_m)
    ax_ts.plot(yr_e, ts_e, "k-o", ms=3, lw=1.5, label="ERA5 ref")
    ax_ts.plot(yr_b, ts_b, "b-s", ms=3, lw=1.5, label="GCM bc")
    if raw_m is not None:
        yr_r, ts_r = _annual_ts(raw_m)
        ax_ts.plot(yr_r, ts_r, "r--^", ms=3, lw=1.2, label="GCM raw")
    ax_ts.set_ylabel(label, fontsize=8); ax_ts.set_title("Annual mean (domain avg)", fontsize=9)
    ax_ts.legend(fontsize=7); ax_ts.grid(True, alpha=0.3); ax_ts.tick_params(labelsize=7)

    # 4. Bias maps
    era5_mean = era5_m.mean("time", skipna=True)
    bc_mean   = bc_m.mean("time", skipna=True)
    bias_bc   = bc_mean - era5_mean

    if raw_m is not None:
        bias_raw = raw_m.mean("time", skipna=True) - era5_mean
        vabs = float(max(np.nanpercentile(np.abs(bias_raw.values), 98),
                         np.nanpercentile(np.abs(bias_bc.values),  98)))
    else:
        bias_raw = None
        vabs = float(np.nanpercentile(np.abs(bias_bc.values), 98))

    proj   = ccrs.PlateCarree()
    extent = [float(era5_m.lon.min()) - 0.5, float(era5_m.lon.max()) + 0.5,
              float(era5_m.lat.min()) - 0.5, float(era5_m.lat.max()) + 0.5]

    def _bias_panel(ax, bias_da, title):
        p = ax.pcolormesh(bias_da.lon, bias_da.lat, bias_da.values,
                          cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                          transform=proj, shading="auto")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3)
        ax.set_extent(extent, crs=proj); ax.set_title(title, fontsize=8)
        return p

    if bias_raw is not None:
        _bias_panel(ax_bias_raw, bias_raw, "Bias: GCM raw − ERA5")
    else:
        ax_bias_raw.set_visible(False)

    p2 = _bias_panel(ax_bias_bc, bias_bc, "Bias: GCM bc − ERA5")

    # shared colorbar in the freed third column
    pos = ax_cb_space.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 + 0.05, 0.015, pos.height * 0.8])
    fig.colorbar(p2, cax=cax, label=f"Δ {label}", extend="both")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path.name)


def plot_validate_summary(results: dict, gcm: str, out_path: Path):
    """2-row summary: monthly climatology + annual time series, all variables."""
    vnames = list(results.keys())
    months = list("JFMAMJJASOND")
    fig, axes = plt.subplots(2, len(vnames), figsize=(5 * len(vnames), 8), sharex="row")
    fig.suptitle(f"Bias-correction summary — {gcm}", fontsize=12)

    for col, vname in enumerate(vnames):
        era5 = results[vname]["era5"]
        bc   = results[vname]["bc"]
        raw  = results[vname]["raw"]
        lbl  = VAR_LABELS.get(vname, vname)

        ax0 = axes[0, col]
        ax0.plot(range(1,13), _monthly_clim(era5), "k-o", ms=3, lw=1.5, label="ERA5")
        ax0.plot(range(1,13), _monthly_clim(bc),   "b-s", ms=3, lw=1.5, label="BC")
        if raw is not None:
            ax0.plot(range(1,13), _monthly_clim(raw), "r--", ms=3, lw=1.2, label="Raw")
        ax0.set_xticks(range(1,13)); ax0.set_xticklabels(months, fontsize=7)
        ax0.set_title(lbl, fontsize=8); ax0.grid(True, alpha=0.3)
        if col == 0:
            ax0.set_ylabel("Monthly mean", fontsize=8); ax0.legend(fontsize=7)

        ax1 = axes[1, col]
        yr_e, ts_e = _annual_ts(era5); yr_b, ts_b = _annual_ts(bc)
        ax1.plot(yr_e, ts_e, "k-o", ms=3, lw=1.5, label="ERA5")
        ax1.plot(yr_b, ts_b, "b-s", ms=3, lw=1.5, label="BC")
        if raw is not None:
            yr_r, ts_r = _annual_ts(raw)
            ax1.plot(yr_r, ts_r, "r--", ms=3, lw=1.2, label="Raw")
        ax1.grid(True, alpha=0.3); ax1.tick_params(labelsize=7)
        if col == 0:
            ax1.set_ylabel("Annual mean", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path.name)


# ── CF / regional plots ───────────────────────────────────────────────────────────

def build_region_mask(gdf, region_col, lons, lats):
    regions = regionmask.from_geopandas(gdf, names=region_col, overlap=False)
    return regions.mask(lons, lats)


def regional_annual_mean(cf_da, mask, gdf, region_col):
    annual = cf_da.resample(time="YE").mean(skipna=True)
    years  = annual.time.dt.year.values
    records = {}
    for i, name in enumerate(gdf[region_col]):
        records[name] = annual.where(mask == i).mean(["lat","lon"], skipna=True).values
    return pd.DataFrame(records, index=years)


def plot_timeseries(dfs, cf_name, gcm, out_path):
    all_states = list(next(iter(dfs.values())).columns)
    ncols = 6
    nrows = int(np.ceil(len(all_states) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.5),
                              sharex=True)
    axes = axes.flatten()
    for i, state in enumerate(all_states):
        ax = axes[i]
        for ssp, df in dfs.items():
            if state in df.columns:
                ax.plot(df.index, df[state], color=SSP_COLORS.get(ssp, "grey"),
                        lw=1.0, label=ssp)
        ax.set_title(state, fontsize=7); ax.tick_params(labelsize=6)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(len(all_states), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{cf_name} annual mean by state — {gcm}", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path.name)


def plot_20y_maps(cf_20y, cf_name, gcm, ssp, gdf, cmap, vmin, vmax, out_path):
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                              subplot_kw={"projection": proj})
    axes = axes.flatten()
    p = None
    for ax, ((start, end), da) in zip(axes, cf_20y.items()):
        p = ax.pcolormesh(da.lon, da.lat, da.values, cmap=cmap,
                          vmin=vmin, vmax=vmax, transform=proj, shading="auto")
        gdf.boundary.plot(ax=ax, color="black", linewidth=0.4, transform=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_title(f"{start}–{end}", fontsize=10)
        ax.set_extent([gdf.total_bounds[0]-0.5, gdf.total_bounds[2]+0.5,
                       gdf.total_bounds[1]-0.5, gdf.total_bounds[3]+0.5], crs=proj)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey",
                          alpha=0.5, linestyle="--")
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {"size": 7}; gl.ylabel_style = {"size": 7}
    fig.colorbar(p, ax=axes, orientation="vertical",
                 fraction=0.02, pad=0.04, shrink=0.8).set_label(cf_name, fontsize=9)
    fig.suptitle(f"{cf_name} 20-year mean — {gcm} {ssp}", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path.name)


# ── Argument parsing ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-dir",      required=True,  type=Path)
    p.add_argument("--cmip-dir",    required=True,  type=Path,
                   help="Raw CMIP6 directory (for raw historical comparison)")
    p.add_argument("--shapefile",   required=True,  type=Path)
    p.add_argument("--out-dir",     required=True,  type=Path)
    p.add_argument("--gcm",         default="CanESM5")
    p.add_argument("--run",         default="r10i1p1f1")
    p.add_argument("--ssps",        nargs="+", default=["ssp245", "ssp585"])
    p.add_argument("--train-start", default="1980-01-01")
    p.add_argument("--train-end",   default="2010-12-31")
    p.add_argument("--region-col",  default="STNAME_SH")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    TRAIN = slice(args.train_start, args.train_end)

    # ══════════════════════════════════════════════════════════════════════════════
    # 0. Validation — raw variable alignment on training period
    # ══════════════════════════════════════════════════════════════════════════════
    log.info("══ Validation plots ══")
    val_results = {}
    for vname in ["tas", "tasmax", "sfcWind", "rsds"]:
        log.info("  %s", vname)
        try:
            era5 = load_era5_ref(args.bc_dir, vname)
            bc   = load_hist_bc(args.bc_dir, vname, args.gcm)
        except FileNotFoundError as e:
            log.warning("  Skipping %s validation — %s", vname, e)
            continue
        raw = load_hist_raw(args.cmip_dir, vname, args.gcm, args.run, TRAIN)
        if raw is not None:
            raw = raw.assign_coords(lat=era5.lat, lon=era5.lon)

        val_results[vname] = {"era5": era5, "bc": bc, "raw": raw}
        plot_validate_var(
            vname, era5, bc, raw, args.gcm,
            args.out_dir / f"validate_{vname}_{args.gcm}.png",
        )

    if val_results:
        plot_validate_summary(
            val_results, args.gcm,
            args.out_dir / f"validate_summary_{args.gcm}.png",
        )
    del val_results

    # ══════════════════════════════════════════════════════════════════════════════
    # 1–3. CF computation, aggregation, and plots (per SSP)
    # ══════════════════════════════════════════════════════════════════════════════
    log.info("══ CF computation ══")
    gdf = gpd.read_file(args.shapefile).to_crs("EPSG:4326")
    log.info("Shapefile: %d regions", len(gdf))

    first_nc = next(args.bc_dir.glob(f"tas_{args.gcm}_ssp*.nc"), None) \
            or next(args.bc_dir.glob("tas_*.nc"))
    with xr.open_dataset(first_nc) as _ds:
        lons, lats = _ds.lon.values, _ds.lat.values

    region_mask = build_region_mask(gdf, args.region_col, lons, lats)

    ts_dfs_wind, ts_dfs_solar = {}, {}

    for ssp in args.ssps:
        log.info("━━━ %s ━━━", ssp)

        sfcWind = load_bc(args.bc_dir, "sfcWind", args.gcm, ssp)
        tas     = load_bc(args.bc_dir, "tas",     args.gcm, ssp)
        tasmax  = load_bc(args.bc_dir, "tasmax",  args.gcm, ssp)
        rsds    = load_bc(args.bc_dir, "rsds",    args.gcm, ssp)

        wCF = wind_cf(sfcWind)
        sCF = solar_cf(tas, tasmax, rsds, sfcWind)
        del sfcWind, tas, tasmax, rsds

        for cf_da, name in [(wCF, "wCF"), (sCF, "sCF")]:
            out_nc = args.out_dir / f"{name}_{args.gcm}_{ssp}.nc"
            if not out_nc.exists():
                cf_da.to_netcdf(out_nc)
                log.info("  → %s", out_nc.name)

        log.info("  Aggregating by region …")
        df_w = regional_annual_mean(wCF, region_mask, gdf, args.region_col)
        df_s = regional_annual_mean(sCF, region_mask, gdf, args.region_col)
        df_w.to_csv(args.out_dir / f"wCF_{args.gcm}_{ssp}_states_annual.csv")
        df_s.to_csv(args.out_dir / f"sCF_{args.gcm}_{ssp}_states_annual.csv")
        ts_dfs_wind[ssp], ts_dfs_solar[ssp] = df_w, df_s

        log.info("  20-year maps …")
        for cf_da, name, cmap, (vmin, vmax) in [
            (wCF, "wCF", "Blues",   (0.0, 0.50)),
            (sCF, "sCF", "Oranges", (0.0, 0.25)),
        ]:
            cf_20y = {
                (s, e): cf_da.sel(time=slice(str(s), str(e))).mean("time", skipna=True).load()
                for s, e in PERIODS
            }
            plot_20y_maps(cf_20y, name, args.gcm, ssp, gdf, cmap, vmin, vmax,
                          args.out_dir / f"map_{name}_{args.gcm}_{ssp}.png")
            del cf_20y

        del wCF, sCF

    log.info("Time-series plots …")
    plot_timeseries(ts_dfs_wind,  "wCF", args.gcm,
                    args.out_dir / f"timeseries_wCF_{args.gcm}.png")
    plot_timeseries(ts_dfs_solar, "sCF", args.gcm,
                    args.out_dir / f"timeseries_sCF_{args.gcm}.png")
    log.info("Done.")


if __name__ == "__main__":
    main()
