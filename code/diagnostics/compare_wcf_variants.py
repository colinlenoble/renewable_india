#!/usr/bin/env python3
"""
Compare the hourly wind capacity factor (wCF) under different turbine assumptions.

Variants (compute_cf.py / compute_wcf_variants.py outputs):
    baseline      wCF_<gcm>_<scn>_hourly.nc               150 m, vci 3.5, vr 13, vco 25
    conservative  wCF_conservative_<gcm>_<scn>_hourly.nc  120 m, vci 3.5, vr 13, vco 25
    optimistic    wCF_optimistic_<gcm>_<scn>_hourly.nc    150 m, vci 3,   vr 11, vco 27

Years are assigned by position (row // 8760) — see analyse_wind_cf_ssp.py.
India and state means are area-weighted by the exact grid cell ∩ polygon overlap
(land only).

Outputs:
    <out-dir>/wCF_variants_<gcm>_states.csv          period means and changes per state
    <out-dir>/wCF_variants_<gcm>_india_annual.csv    India annual mean per variant/scenario
    <fig-dir>/wcf_variants_india_<gcm>.png           annual series + seasonal cycle + hourly distribution
    <fig-dir>/wcf_variants_maps_<gcm>.png            baseline-period maps + optimistic/conservative ratio
    <fig-dir>/wcf_variants_change_<gcm>.png          end-of-century change maps per variant
    <fig-dir>/wcf_variants_states_<gcm>.png          per-state level and change, per variant

Usage
-----
python compare_wcf_variants.py --cf-dir E:/renewable_india/CanESM5 --gcm CanESM5
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
from shapely.geometry import box

HOURS_PER_YEAR = 8760
START_YEAR = {"historical": 1980, "ssp126": 2015, "ssp245": 2015, "ssp370": 2015, "ssp585": 2015}
BASELINE = (1981, 2010)
FUTURE = {"2021-2050": (2021, 2050), "2071-2100": (2071, 2100)}
VARIANTS = {"conservative": "conservative_", "baseline": "", "optimistic": "optimistic_"}
VLABEL = {"conservative": "Conservative (120 m, vr 13)",
          "baseline": "Current (150 m, vr 13)",
          "optimistic": "Optimistic (150 m, vr 11, vci 3, vco 27)"}
VCOLOR = {"conservative": "#2a78d6", "baseline": "#52514e", "optimistic": "#eb6834"}
SLABEL = {"historical": "Historical", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
SSTYLE = {"historical": "-", "ssp245": "--", "ssp585": "-"}
MAIN_WIND_STATES = ["Gujarat", "Tamil Nadu", "Karnataka", "Maharashtra",
                    "Rajasthan", "Andhra Pradesh", "Madhya Pradesh", "Telangana"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32b2b"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
})


# ---------------------------------------------------------------- data
def overlap_weights(lat, lon, gdf, region_col):
    """Area of each cell ∩ each region (deg² × cos(lat) ∝ true area)."""
    dlat, dlon = np.diff(lat).mean() / 2, np.diff(lon).mean() / 2
    regions = gdf.dissolve(region_col).geometry
    w = np.zeros((len(regions), lat.size, lon.size))
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            cell = box(lo - dlon, la - dlat, lo + dlon, la + dlat)
            w[:, i, j] = np.array([g.intersection(cell).area for g in regions]) * np.cos(np.deg2rad(la))
    return xr.DataArray(w, coords={"region": regions.index.values, "lat": lat, "lon": lon},
                        dims=("region", "lat", "lon"))


def summarise(path, scenario, w_india):
    """Annual-mean maps, India hourly series, monthly cycle and hourly-value shares."""
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        cf = ds["wCF"].values.astype(np.float32)          # (time, lat, lon), ~330 MB
        lat, lon = ds.lat.values, ds.lon.values
        month = pd.DatetimeIndex(ds.time.values).month    # calendar drift < 1 month
    n_years = cf.shape[0] // HOURS_PER_YEAR
    years = START_YEAR[scenario] + np.arange(n_years)
    annual = cf.reshape(n_years, HOURS_PER_YEAR, *cf.shape[1:]).mean(1)
    wi = np.nan_to_num(w_india.values)
    india_h = np.nansum(cf * wi, axis=(1, 2)) / wi.sum()
    land = wi > 0
    share_zero = (cf[:, land] == 0).reshape(n_years, HOURS_PER_YEAR, -1).mean((1, 2))
    share_rated = (cf[:, land] >= 0.999).reshape(n_years, HOURS_PER_YEAR, -1).mean((1, 2))
    out = {
        "annual": xr.DataArray(annual, coords={"year": years, "lat": lat, "lon": lon},
                               dims=("year", "lat", "lon")),
        "india_hourly": pd.Series(india_h, index=np.repeat(years, HOURS_PER_YEAR)),
        "monthly": pd.Series(india_h).groupby(np.asarray(month)).mean(),
        "share_zero": pd.Series(share_zero, index=years),
        "share_rated": pd.Series(share_rated, index=years),
        "hourly_land": cf[:, land],                      # kept only for the histogram periods
    }
    return out


def period(series_or_da, a, b, dim="year"):
    if isinstance(series_or_da, xr.DataArray):
        return series_or_da.sel({dim: slice(a, b)}).mean(dim)
    return series_or_da.loc[a:b].mean()


# ---------------------------------------------------------------- figures
def style_map(ax):
    ax.set_xlim(67, 98); ax.set_ylim(6, 37.5); ax.set_aspect("equal"); ax.grid(False)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_india(res, india_annual, hist_dist, gcm, out):
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.35, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    for v in VARIANTS:
        for scn in res[v]:
            s = india_annual[(v, scn)]
            roll = s.rolling(11, center=True, min_periods=6).mean()
            ax.plot(s.index, s.values, color=VCOLOR[v], lw=0.7, alpha=0.25, ls=SSTYLE[scn])
            ax.plot(roll.index, roll.values, color=VCOLOR[v], lw=2 if scn != "ssp245" else 1.4,
                    ls=SSTYLE[scn])
        ax.text(2101.5, india_annual[(v, "ssp585")].rolling(11, center=True, min_periods=6).mean().dropna().iloc[-1],
                VLABEL[v].split(" (")[0], color=VCOLOR[v], va="center", fontsize=8)
    ax.plot([], [], color=MUTED, ls="-", lw=2, label="Historical / SSP5-8.5")
    ax.plot([], [], color=MUTED, ls="--", lw=1.4, label="SSP2-4.5")
    ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=2)
    ax.set_xlim(1978, 2113)
    ax.set_ylabel("Annual mean wind CF (–)")
    ax.set_title(f"India land-area mean wind CF by turbine assumption ({gcm}) — annual (thin) and 11-yr mean (thick)",
                 loc="left", fontsize=10, color=INK)

    ax = fig.add_subplot(gs[1, 0])
    for v in VARIANTS:
        m = res[v]["historical"]["monthly"]
        ax.plot(m.index, m.values, color=VCOLOR[v], lw=2, marker="o", ms=3, label=VLABEL[v].split(" (")[0])
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_ylabel("Mean wind CF (–)")
    ax.set_title(f"Seasonal cycle, {BASELINE[0]}-{BASELINE[1]}", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 21)
    for v in VARIANTS:
        x = hist_dist[v]
        h, _ = np.histogram(x[x > 0], bins=bins)
        ax.step(bins[:-1], h / x.size * 100, where="post", color=VCOLOR[v], lw=1.6)
    ax.set_xlabel("Hourly wind CF, land pixels (CF > 0)")
    ax.set_ylabel("Share of pixel-hours (%)")
    ax.set_title(f"Hourly distribution, {BASELINE[0]}-{BASELINE[1]}", loc="left", fontsize=10, color=INK)

    ax = fig.add_subplot(gs[1, 2])
    cats = ["CF = 0", "0 < CF < 1", "CF = 1"]
    x = np.arange(3)
    for k, v in enumerate(VARIANTS):
        d = hist_dist[v]
        z, r = (d == 0).mean(), (d >= 0.999).mean()
        vals = np.array([z, 1 - z - r, r]) * 100
        ax.bar(x + (k - 1) * 0.27, vals, width=0.25, color=VCOLOR[v])
        for xi, val in zip(x + (k - 1) * 0.27, vals):
            ax.text(xi, val + 1, f"{val:.0f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, cats)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Share of land pixel-hours (%)")
    ax.set_title("Time below cut-in / at rated power", loc="left", fontsize=10, color=INK)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_maps(base_maps, gdf_r, mask, gcm, out):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    vmax = float(max(np.nanmax(m.where(mask)) for m in base_maps.values()))
    for ax, (v, m) in zip(axes, base_maps.items()):
        im = ax.pcolormesh(m.lon, m.lat, m.where(mask), cmap="viridis", vmin=0, vmax=vmax,
                           shading="nearest", edgecolors="white", lw=0.3)
        gdf_r.boundary.plot(ax=ax, color="white", lw=1.2)
        gdf_r.boundary.plot(ax=ax, color=INK, lw=0.4)
        ax.set_title(VLABEL[v], loc="left", fontsize=9, color=INK)
        style_map(ax)
    fig.colorbar(im, ax=axes[:3], shrink=0.75, pad=0.01).set_label(
        f"Mean wind CF {BASELINE[0]}-{BASELINE[1]} (–)")
    # ratio explodes where the conservative CF is ~0 (Himalaya, far north-east): mask those
    ratio = base_maps["optimistic"] / base_maps["conservative"]
    ratio = ratio.where(mask & (base_maps["conservative"] >= 0.02))
    ax = axes[3]
    im = ax.pcolormesh(ratio.lon, ratio.lat, ratio, cmap="magma_r", vmin=1,
                       vmax=float(np.nanmax(ratio)), shading="nearest", edgecolors="white", lw=0.3)
    gdf_r.boundary.plot(ax=ax, color="white", lw=1.2)
    gdf_r.boundary.plot(ax=ax, color=INK, lw=0.4)
    ax.set_title("Optimistic ÷ conservative (CF ≥ 0.02)", loc="left", fontsize=9, color=INK)
    style_map(ax)
    fig.colorbar(im, ax=axes[3], shrink=0.75, pad=0.02).set_label("Ratio (–)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_change(res, gdf_r, mask, gcm, out):
    rows = ["ssp245", "ssp585"]
    a, b = FUTURE["2071-2100"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.6))
    rel = {}
    for r, scn in enumerate(rows):
        for c, v in enumerate(VARIANTS):
            base = period(res[v]["historical"]["annual"], *BASELINE)
            fut = period(res[v][scn]["annual"], a, b)
            rel[(scn, v)] = ((fut - base) / base * 100).where(mask & (base > 0.02))
    vmax = float(np.nanpercentile(np.abs(np.stack([d.values for d in rel.values()])), 98))
    norm = TwoSlopeNorm(0, -vmax, vmax)
    for (scn, v), d in rel.items():
        ax = axes[rows.index(scn), list(VARIANTS).index(v)]
        im = ax.pcolormesh(d.lon, d.lat, d, cmap=DIVERGING, norm=norm, shading="nearest",
                           edgecolors="white", lw=0.3)
        gdf_r.boundary.plot(ax=ax, color=INK, lw=0.4)
        ax.set_title(f"{SLABEL[scn]} — {VLABEL[v].split(' (')[0]}", loc="left", fontsize=9, color=INK)
        style_map(ax)
    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, extend="both").set_label(
        f"Change in mean wind CF, {a}-{b} vs {BASELINE[0]}-{BASELINE[1]} (%)\n"
        "pixels with baseline CF < 0.02 masked")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_states(tab, gcm, out):
    st = [s for s in MAIN_WIND_STATES if s in tab.index]
    t = tab.loc[st]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    y = np.arange(len(st))
    h = 0.27
    cols = [("base", f"Mean wind CF {BASELINE[0]}-{BASELINE[1]} (–)"),
            ("d%_ssp245_2071-2100", "Change 2071-2100, SSP2-4.5 (%)"),
            ("d%_ssp585_2071-2100", "Change 2071-2100, SSP5-8.5 (%)")]
    for ax, (col, xl) in zip(axes, cols):
        for k, v in enumerate(VARIANTS):
            ax.barh(y + (1 - k) * h, t[(v, col)], height=h - 0.03, color=VCOLOR[v],
                    label=VLABEL[v].split(" (")[0])
        ax.axvline(0, color="#c3c2b7", lw=1)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, st, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle(f"Main wind states by turbine assumption ({gcm})", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=3,
               loc="upper right", bbox_to_anchor=(0.99, 1.02))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------- main
def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--cf-dir", type=Path, default=Path("E:/renewable_india/CanESM5"))
    p.add_argument("--gcm", default="CanESM5")
    p.add_argument("--scenarios", nargs="+", default=["historical", "ssp245", "ssp585"])
    p.add_argument("--shapefile", type=Path, default=root / "INDIA_STATES.geojson")
    p.add_argument("--region-col", default="STNAME_SH")
    p.add_argument("--out-dir", type=Path, default=root / "data" / "proc" / "wind_cf_variants")
    p.add_argument("--fig-dir", type=Path, default=root / "figures" / "wind_cf_variants")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(args.shapefile).to_crs("EPSG:4326")
    gdf_r = gdf.dissolve(args.region_col).reset_index()
    first = args.cf_dir / f"wCF_{args.gcm}_historical_hourly.nc"
    with xr.open_dataset(first, engine="h5netcdf") as ds:
        lat, lon = ds.lat.values, ds.lon.values
    w = overlap_weights(lat, lon, gdf, args.region_col)
    w_india = w.sum("region")
    mask = w_india > 0

    res, hist_dist = {}, {}
    for v, prefix in VARIANTS.items():
        res[v] = {}
        for scn in args.scenarios:
            path = args.cf_dir / f"wCF_{prefix}{args.gcm}_{scn}_hourly.nc"
            print(f"Reading {path.name} …")
            r = summarise(path, scn, w_india)
            if scn == "historical":
                y0 = START_YEAR[scn]
                hist_dist[v] = r["hourly_land"][(BASELINE[0] - y0) * HOURS_PER_YEAR:
                                                (BASELINE[1] - y0 + 1) * HOURS_PER_YEAR].ravel()
            del r["hourly_land"]
            res[v][scn] = r

    # hourly ordering check: conservative <= baseline <= optimistic
    print("\nHourly ordering (historical, land pixel-hours):")
    for lo_v, hi_v in [("conservative", "baseline"), ("baseline", "optimistic")]:
        viol = (hist_dist[lo_v] > hist_dist[hi_v] + 1e-6).mean()
        print(f"  {lo_v} > {hi_v}: {viol * 100:.4f} % of hours")

    # India annual means
    india_annual = {}
    for v in VARIANTS:
        for scn in args.scenarios:
            s = res[v][scn]["india_hourly"]
            india_annual[(v, scn)] = s.groupby(level=0).mean()
    pd.DataFrame({f"{v}|{s}": x for (v, s), x in india_annual.items()}).to_csv(
        args.out_dir / f"wCF_variants_{args.gcm}_india_annual.csv")

    # India summary
    print("\nIndia (land area-weighted) mean wind CF:")
    rows = []
    for v in VARIANTS:
        base = period(india_annual[(v, "historical")], *BASELINE)
        row = {"variant": v, f"{BASELINE[0]}-{BASELINE[1]}": round(base, 4),
               "share CF=0 (%)": round(period(res[v]["historical"]["share_zero"], *BASELINE) * 100, 1),
               "share CF=1 (%)": round(period(res[v]["historical"]["share_rated"], *BASELINE) * 100, 1)}
        for scn in [s for s in args.scenarios if s != "historical"]:
            for per, (a, b) in FUTURE.items():
                f = period(india_annual[(v, scn)], a, b)
                row[f"{scn} {per} (%)"] = round((f - base) / base * 100, 1)
        rows.append(row)
    india_tab = pd.DataFrame(rows).set_index("variant")
    print(india_tab.to_string())
    india_tab.to_csv(args.out_dir / f"wCF_variants_{args.gcm}_india_summary.csv")

    # State table
    wsum = w.sum(("lat", "lon"))
    cols = {}
    for v in VARIANTS:
        base = (w * period(res[v]["historical"]["annual"], *BASELINE)).sum(("lat", "lon")) / wsum
        cols[(v, "base")] = base.to_series()
        for scn in [s for s in args.scenarios if s != "historical"]:
            for per, (a, b) in FUTURE.items():
                fut = (w * period(res[v][scn]["annual"], a, b)).sum(("lat", "lon")) / wsum
                cols[(v, f"d%_{scn}_{per}")] = ((fut - base) / base * 100).to_series()
    tab = pd.DataFrame(cols)
    tab[("ratio", "opt/cons")] = tab[("optimistic", "base")] / tab[("conservative", "base")]
    tab = tab.sort_values(("baseline", "base"), ascending=False)
    tab.round(4).to_csv(args.out_dir / f"wCF_variants_{args.gcm}_states.csv")
    show = tab.loc[[s for s in MAIN_WIND_STATES if s in tab.index]]
    print("\nMain wind states:")
    print(show[[(v, "base") for v in VARIANTS] + [("ratio", "opt/cons")]
               + [(v, "d%_ssp585_2071-2100") for v in VARIANTS]].round(3).to_string())

    base_maps = {v: period(res[v]["historical"]["annual"], *BASELINE) for v in VARIANTS}
    fig_india(res, india_annual, hist_dist, args.gcm, args.fig_dir / f"wcf_variants_india_{args.gcm}.png")
    fig_maps(base_maps, gdf_r, mask, args.gcm, args.fig_dir / f"wcf_variants_maps_{args.gcm}.png")
    fig_change(res, gdf_r, mask, args.gcm, args.fig_dir / f"wcf_variants_change_{args.gcm}.png")
    fig_states(tab, args.gcm, args.fig_dir / f"wcf_variants_states_{args.gcm}.png")


if __name__ == "__main__":
    main()
