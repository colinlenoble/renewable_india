#!/usr/bin/env python3
"""
Annual-mean wind capacity factor (wCF) per SSP: evolution across years.

Inputs (compute_cf.py / aggregate_cf_states.py outputs):
    <cf-dir>/wCF_<gcm>_<scenario>_states_hourly.csv   (state means, area-weighted by xagg)
    <cf-dir>/wCF_<gcm>_<scenario>_hourly.nc           (gridded, for the change map)

Calendar note
-------------
The GCM runs on a 365-day (noleap) calendar but the time axis was written as
consecutive standard datetimes, so the stamps drift (historical ends on
2010-12-23 instead of 12-31). Every series is an exact multiple of 8760 h, so
years are assigned by position (row // 8760) rather than by `time.year`.

India-wide mean = state means weighted by state area (equal-area projection) —
land only, unlike a plain box mean over the grid which includes the sea.

Outputs:
    <out-dir>/wCF_<gcm>_annual_states.csv        annual mean per state + India
    <fig-dir>/wcf_india_annual_<gcm>.png         India annual mean, all scenarios
    <fig-dir>/wcf_states_timeseries_<gcm>.png    small multiples, main wind states
    <fig-dir>/wcf_states_change_<gcm>.png        per-state change vs 1981-2010
    <fig-dir>/wcf_change_map_<gcm>.png           gridded change 2071-2100 vs 1981-2010

Usage
-----
python analyse_wind_cf_ssp.py --cf-dir E:/renewable_india --gcm GFDL-ESM4
"""

import argparse
import os
import sys
from pathlib import Path

_proj_dir = Path(sys.prefix) / "Library" / "share" / "proj"
if _proj_dir.exists():
    os.environ["PROJ_DATA"] = str(_proj_dir)
    os.environ["PROJ_LIB"] = str(_proj_dir)

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats
from shapely.geometry import box

HOURS_PER_YEAR = 8760
START_YEAR = {"historical": 1980, "ssp126": 2015, "ssp370": 2015}
BASELINE = (1981, 2010)
PERIODS = {"2021-2050": (2021, 2050), "2041-2070": (2041, 2070), "2071-2100": (2071, 2100)}
MAIN_WIND_STATES = ["Gujarat", "Tamil Nadu", "Karnataka", "Maharashtra",
                    "Rajasthan", "Andhra Pradesh", "Madhya Pradesh", "Telangana"]

# Palette: neutral ink for historical, validated categorical slots 1-2 for SSPs
COLORS = {"historical": "#52514e", "ssp126": "#2a78d6", "ssp370": "#eb6834"}
LABELS = {"historical": "Historical", "ssp126": "SSP1-2.6", "ssp370": "SSP3-7.0"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32b2b"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
})


def year_index(n_rows, scenario):
    if n_rows % HOURS_PER_YEAR:
        raise ValueError(f"{scenario}: {n_rows} rows is not a whole number of 365-day years")
    return START_YEAR[scenario] + np.arange(n_rows) // HOURS_PER_YEAR


def annual_states(cf_dir, gcm, scenario):
    path = cf_dir / f"wCF_{gcm}_{scenario}_states_hourly.csv"
    print(f"Reading {path.name} …")
    df = pd.read_csv(path, index_col=0)
    df = df.astype(np.float32)
    df.index = year_index(len(df), scenario)
    return df.groupby(level=0).mean()


def state_areas(shapefile, states, region_col):
    gdf = gpd.read_file(shapefile).to_crs("EPSG:6933")   # equal-area
    area = gdf.set_index(region_col).geometry.area
    return area.groupby(level=0).sum().reindex(states)


def annual_grid(cf_dir, gcm, scenario):
    path = cf_dir / f"wCF_{gcm}_{scenario}_hourly.nc"
    print(f"Reading {path.name} …")
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        n_years = ds.sizes["time"] // HOURS_PER_YEAR
        out = np.empty((n_years, ds.sizes["lat"], ds.sizes["lon"]), np.float32)
        for i in range(n_years):
            out[i] = ds["wCF"].isel(time=slice(i * HOURS_PER_YEAR, (i + 1) * HOURS_PER_YEAR)).values.mean(0)
        years = START_YEAR[scenario] + np.arange(n_years)
        return xr.DataArray(out, coords={"year": years, "lat": ds.lat.values, "lon": ds.lon.values},
                            dims=("year", "lat", "lon"), name="wCF")


def land_fraction(lat, lon, shapefile):
    """Fraction of each grid cell covered by Indian states (coarse 1°x1.25° grid)."""
    union = gpd.read_file(shapefile).to_crs("EPSG:4326").union_all()
    dlat, dlon = np.diff(lat).mean() / 2, np.diff(lon).mean() / 2
    frac = np.zeros((lat.size, lon.size))
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            cell = box(lo - dlon, la - dlat, lo + dlon, la + dlat)
            frac[i, j] = cell.intersection(union).area / cell.area
    return frac


def period_mean(s, period):
    return s.loc[period[0]:period[1]].mean()


def trend(s):
    """Theil-Sen slope (per decade) + Mann-Kendall-style p-value (Kendall tau vs year)."""
    s = s.dropna()
    slope, _, lo, hi = stats.theilslopes(s.values, s.index.values)
    p = stats.kendalltau(s.index.values, s.values).pvalue
    return slope * 10, lo * 10, hi * 10, p


# ---------------------------------------------------------------- figures
def fig_india(annual, fig_dir, gcm):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    base = period_mean(annual["historical"]["India"], BASELINE)
    ax.axhline(base, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(1980, base + 0.0001, f"1981-2010 mean {base:.4f}", color=MUTED, va="bottom", fontsize=7,
            bbox=dict(fc="white", ec="none", pad=0.5))
    for scn, df in annual.items():
        s = df["India"]
        ax.plot(s.index, s.values, color=COLORS[scn], lw=1, alpha=0.45)
        ax.scatter(s.index, s.values, color=COLORS[scn], s=9, alpha=0.6, lw=0)
        roll = s.rolling(11, center=True, min_periods=6).mean()
        ax.plot(roll.index, roll.values, color=COLORS[scn], lw=2.2, label=f"{LABELS[scn]} (11-yr mean)")
        if scn != "historical":
            ax.text(s.index[-1] + 1, roll.dropna().iloc[-1], LABELS[scn],
                    color=INK, va="center", fontsize=8)
    ax.set_xlim(1978, 2108)
    ax.set_ylabel("Annual mean wind CF (–)")
    ax.set_title(f"India: annual-mean wind capacity factor ({gcm}, land area-weighted)",
                 loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    out = fig_dir / f"wcf_india_annual_{gcm}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_states_ts(annual, fig_dir, gcm):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    for ax, st in zip(axes.flat, MAIN_WIND_STATES):
        for scn, df in annual.items():
            s = df[st]
            ax.plot(s.index, s.values, color=COLORS[scn], lw=0.8, alpha=0.35)
            roll = s.rolling(11, center=True, min_periods=6).mean()
            ax.plot(roll.index, roll.values, color=COLORS[scn], lw=1.8, label=LABELS[scn])
        ax.axhline(period_mean(annual["historical"][st], BASELINE), color=MUTED, lw=0.8, ls="--")
        ax.set_title(st, loc="left", fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("Annual mean wind CF")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Main wind states — annual mean (thin) and 11-yr mean (thick), {gcm}",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    out = fig_dir / f"wcf_states_timeseries_{gcm}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_states_change(change, base, fig_dir, gcm):
    # drop states with negligible baseline wind resource — relative changes meaningless there
    keep = base[base >= 0.02].index
    ch = change.loc[keep].sort_values(("ssp370", "2071-2100"))
    y = np.arange(len(ch))
    fig, ax = plt.subplots(figsize=(7.5, 0.26 * len(ch) + 1.4))
    h = 0.38
    for k, scn in enumerate(["ssp126", "ssp370"]):
        ax.barh(y + (k - 0.5) * h, ch[(scn, "2071-2100")] * 100, height=h - 0.04,
                color=COLORS[scn], label=LABELS[scn])
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_yticks(y, [f"{s}  ({base[s]:.2f})" for s in ch.index], fontsize=8, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in annual-mean wind CF, 2071-2100 vs 1981-2010 (percentage points)")
    ax.set_title(f"Change by state ({gcm}); baseline CF in brackets",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = fig_dir / f"wcf_states_change_{gcm}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_change_map(grid, frac, shapefile, fig_dir, gcm):
    gdf = gpd.read_file(shapefile).to_crs("EPSG:4326")
    base = grid["historical"].sel(year=slice(*BASELINE)).mean("year")
    diffs = {scn: (grid[scn].sel(year=slice(2071, 2100)).mean("year") - base) * 100
             for scn in ["ssp126", "ssp370"]}
    mask = frac < 0.25
    # 98th percentile, not max: a single coastal cell (Sundarbans) otherwise flattens the scale
    vmax = float(np.nanpercentile(np.abs(np.stack([d.where(~mask).values for d in diffs.values()])), 98))
    norm = TwoSlopeNorm(0, -vmax, vmax)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, (scn, d) in zip(axes, diffs.items()):
        im = ax.pcolormesh(d.lon, d.lat, d.where(~mask), cmap=DIVERGING, norm=norm, shading="nearest")
        gdf.boundary.plot(ax=ax, color="#52514e", lw=0.4)
        ax.set_title(f"{LABELS[scn]}: 2071-2100 minus 1981-2010", loc="left", fontsize=10, color=INK)
        ax.set_aspect("equal"); ax.grid(False)
        ax.set_xlim(67, 98); ax.set_ylim(6, 37.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, extend="both")
    cb.set_label("Δ annual-mean wind CF (percentage points)")
    out = fig_dir / f"wcf_change_map_{gcm}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--cf-dir", type=Path, default=Path("E:/renewable_india"))
    p.add_argument("--gcm", default="GFDL-ESM4")
    p.add_argument("--scenarios", nargs="+", default=["historical", "ssp126", "ssp370"])
    root = Path(__file__).resolve().parents[2]
    p.add_argument("--shapefile", type=Path, default=root / "INDIA_STATES.geojson")
    p.add_argument("--region-col", default="STNAME_SH")
    p.add_argument("--out-dir", type=Path, default=root / "data" / "proc" / "wind_cf_annual")
    p.add_argument("--fig-dir", type=Path, default=root / "figures" / "wind_cf_ssp")
    p.add_argument("--no-map", action="store_true", help="skip the gridded change map (reads ~5 GB)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    annual = {scn: annual_states(args.cf_dir, args.gcm, scn) for scn in args.scenarios}
    states = annual[args.scenarios[0]].columns
    w = state_areas(args.shapefile, states, args.region_col)
    if w.isna().any():
        raise ValueError(f"No area for states: {list(w[w.isna()].index)}")
    for df in annual.values():
        df["India"] = (df[states] * w.values).sum(axis=1) / w.sum()

    long = pd.concat({LABELS[s]: df for s, df in annual.items()}, names=["scenario", "year"])
    csv = args.out_dir / f"wCF_{args.gcm}_annual_states.csv"
    long.to_csv(csv)
    print(f"  -> {csv}")

    # ---- summary: period means, change vs baseline, trends
    base = annual["historical"].loc[BASELINE[0]:BASELINE[1]].mean()
    change = pd.DataFrame({(scn, per): annual[scn].loc[a:b].mean() - base
                           for scn in args.scenarios if scn != "historical"
                           for per, (a, b) in PERIODS.items()})
    print(f"\nIndia annual-mean wCF, baseline 1981-2010 = {base['India']:.4f}")
    rows = []
    for scn in [s for s in args.scenarios if s != "historical"]:
        sl, lo, hi, pv = trend(annual[scn]["India"].loc[2015:2100])
        row = {"scenario": LABELS[scn],
               "trend 2015-2100 (pp/decade)": f"{sl*100:+.3f} [{lo*100:+.3f}, {hi*100:+.3f}]",
               "Kendall p": f"{pv:.3g}"}
        for per in PERIODS:
            d = change.loc["India", (scn, per)]
            row[f"Δ {per} (pp / %)"] = f"{d*100:+.2f} / {d/base['India']*100:+.1f}%"
        rows.append(row)
    sl, lo, hi, pv = trend(annual["historical"]["India"])
    rows.append({"scenario": "Historical", "trend 2015-2100 (pp/decade)":
                 f"(1980-2010) {sl*100:+.3f} [{lo*100:+.3f}, {hi*100:+.3f}]", "Kendall p": f"{pv:.3g}"})
    summary = pd.DataFrame(rows).set_index("scenario")
    print(summary.to_string())
    summary.to_csv(args.out_dir / f"wCF_{args.gcm}_india_summary.csv")

    rel = (change.xs("2071-2100", axis=1, level=1).div(base, axis=0) * 100).round(1)
    tab = pd.concat({"baseline": base.round(4),
                     "Δpp ssp126": (change[("ssp126", "2071-2100")] * 100).round(2),
                     "Δpp ssp370": (change[("ssp370", "2071-2100")] * 100).round(2),
                     "Δ% ssp126": rel["ssp126"], "Δ% ssp370": rel["ssp370"]}, axis=1)
    tab = tab.sort_values("baseline", ascending=False)
    tab.to_csv(args.out_dir / f"wCF_{args.gcm}_state_change_2071-2100.csv")
    print("\nState change 2071-2100 vs 1981-2010:\n" + tab.to_string())

    fig_india(annual, args.fig_dir, args.gcm)
    fig_states_ts(annual, args.fig_dir, args.gcm)
    fig_states_change(change, base.drop("India"), args.fig_dir, args.gcm)

    if not args.no_map:
        grid = {scn: annual_grid(args.cf_dir, args.gcm, scn) for scn in args.scenarios}
        frac = land_fraction(grid["historical"].lat.values, grid["historical"].lon.values, args.shapefile)
        fig_change_map(grid, frac, args.shapefile, args.fig_dir, args.gcm)


if __name__ == "__main__":
    main()
