#!/usr/bin/env python3
"""
Maps of period-mean wind capacity factor (wCF), by region and by pixel.

Row 1: state means (area-weighted by the exact cell ∩ polygon overlap), with the
       GCM grid drawn on top so you can see which pixels feed which state.
Row 2: the per-pixel period means, with state boundaries on top.
Columns: the requested periods (default 2020-2040 and 2060-2080).

Years are assigned by position (row // 8760) from the scenario start year —
see analyse_wind_cf_ssp.py for the noleap calendar note.

Usage
-----
python map_wind_cf_periods.py --gcm GFDL-ESM4 --scenario ssp126
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
from shapely.geometry import box

HOURS_PER_YEAR = 8760
START_YEAR = {"historical": 1980, "ssp126": 2015, "ssp245": 2015, "ssp370": 2015, "ssp585": 2015}
INK, MUTED = "#0b0b0b", "#898781"
CMAP = "viridis"


def period_means(path, scenario, periods):
    """Per-pixel mean wCF over each (start, end) year period, inclusive."""
    out = {}
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        n_years = ds.sizes["time"] // HOURS_PER_YEAR
        y0 = START_YEAR[scenario]
        for a, b in periods:
            i0, i1 = a - y0, b - y0 + 1
            if i0 < 0 or i1 > n_years:
                raise ValueError(f"{a}-{b} outside {y0}-{y0 + n_years - 1}")
            acc = np.zeros((ds.sizes["lat"], ds.sizes["lon"]), np.float64)
            for i in range(i0, i1):   # one year at a time keeps memory at ~25 MB
                acc += ds["wCF"].isel(time=slice(i * HOURS_PER_YEAR, (i + 1) * HOURS_PER_YEAR)).values.mean(0)
            out[(a, b)] = xr.DataArray((acc / (i1 - i0)).astype(np.float32),
                                       coords={"lat": ds.lat.values, "lon": ds.lon.values},
                                       dims=("lat", "lon"))
    return out


def overlap_weights(lat, lon, gdf, region_col):
    """Area of each cell ∩ each region (deg² × cos(lat) ∝ true area at this cell size)."""
    dlat, dlon = np.diff(lat).mean() / 2, np.diff(lon).mean() / 2
    regions = gdf.dissolve(region_col).geometry
    w = np.zeros((len(regions), lat.size, lon.size))
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            cell = box(lo - dlon, la - dlat, lo + dlon, la + dlat)
            w[:, i, j] = np.array([g.intersection(cell).area for g in regions]) * np.cos(np.deg2rad(la))
    return xr.DataArray(w, coords={"region": regions.index.values, "lat": lat, "lon": lon},
                        dims=("region", "lat", "lon"))


def grid_edges(c):
    d = np.diff(c).mean()
    return np.concatenate([c - d / 2, [c[-1] + d / 2]])


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--cf-dir", type=Path, default=Path("E:/renewable_india"))
    p.add_argument("--gcm", default="GFDL-ESM4")
    p.add_argument("--scenario", default="ssp126")
    p.add_argument("--periods", nargs="+", default=["2020-2040", "2060-2080"])
    p.add_argument("--shapefile", type=Path, default=root / "INDIA_STATES.geojson")
    p.add_argument("--region-col", default="STNAME_SH")
    p.add_argument("--fig-dir", type=Path, default=root / "figures" / "wind_cf_ssp")
    p.add_argument("--out-dir", type=Path, default=root / "data" / "proc" / "wind_cf_annual")
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    periods = [tuple(int(y) for y in s.split("-")) for s in args.periods]
    path = args.cf_dir / args.gcm / args.scenario / f"wCF_{args.gcm}_{args.scenario}_hourly.nc"
    print(f"Reading {path} …")
    pix = period_means(path, args.scenario, periods)

    gdf = gpd.read_file(args.shapefile).to_crs("EPSG:4326")
    lat, lon = pix[periods[0]].lat.values, pix[periods[0]].lon.values
    w = overlap_weights(lat, lon, gdf, args.region_col)
    reg = {per: (w * m).sum(("lat", "lon")) / w.sum(("lat", "lon")) for per, m in pix.items()}

    table = pd.DataFrame({f"{a}-{b}": r.to_series() for (a, b), r in reg.items()})
    table["n_pixels"] = (w > 0).sum(("lat", "lon")).to_series()
    table = table.sort_values(table.columns[0], ascending=False)
    csv = args.out_dir / f"wCF_{args.gcm}_{args.scenario}_period_means_states.csv"
    table.round(4).to_csv(csv)
    print(table.round(4).to_string())
    print(f"  -> {csv}")

    # pixels touching India (any overlap) are shown; the rest is masked
    india = w.sum("region") > 0
    gdf_r = gdf.dissolve(args.region_col).reset_index()
    vals = np.concatenate([m.where(india).values.ravel() for m in pix.values()])
    vmin, vmax = np.nanpercentile(vals, [1, 99])
    xe, ye = grid_edges(lon), grid_edges(lat)
    cells = [box(xe[j], ye[i], xe[j + 1], ye[i + 1]) for i, j in zip(*np.nonzero(india.values))]

    n = len(periods)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 10.4))
    axes = np.atleast_2d(axes).reshape(2, n)
    for k, per in enumerate(periods):
        label = f"{per[0]}-{per[1]}"
        ax = axes[0, k]
        g = gdf_r.assign(wCF=gdf_r[args.region_col].map(reg[per].to_series()))
        g.plot(column="wCF", ax=ax, cmap=CMAP, vmin=vmin, vmax=vmax, edgecolor="white", lw=0.4)
        gpd.GeoSeries(cells).boundary.plot(ax=ax, color=INK, lw=0.4, alpha=0.6)
        ax.set_title(f"By state — {label}", loc="left", fontsize=10, color=INK)

        ax = axes[1, k]
        im = ax.pcolormesh(xe, ye, pix[per].where(india), cmap=CMAP, vmin=vmin, vmax=vmax,
                           edgecolors="white", lw=0.2)
        gdf_r.boundary.plot(ax=ax, color="white", lw=1.4)
        gdf_r.boundary.plot(ax=ax, color=INK, lw=0.6)
        ax.set_title(f"By pixel — {label}", loc="left", fontsize=10, color=INK)

    for ax in axes.flat:
        ax.set_xlim(67, 98); ax.set_ylim(6, 37.5)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7, colors=MUTED)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, extend="both")
    cb.set_label("Mean wind capacity factor (–)")
    fig.suptitle(f"Wind capacity factor, {args.gcm} {args.scenario}", x=0.06, ha="left",
                 fontsize=12, color=INK)
    out = args.fig_dir / f"wcf_period_maps_{args.gcm}_{args.scenario}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
