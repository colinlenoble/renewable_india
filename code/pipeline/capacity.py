#!/usr/bin/env python3
"""
Installed-capacity weighting shared by the pipeline steps.

  * GEM Global Wind / Solar Power Tracker reading (operating projects in India)
  * capacity map per pixel of any regular lat/lon grid
  * capacity weights per (region, pixel) for state aggregation
  * WeightedConservativeRegridder: ERA5 0.25° → GCM grid, each GCM cell being
    the capacity-weighted mean of the ERA5 pixels it overlaps

Capacity jitter
---------------
A small floor ``jitter_mw`` (MW per pixel of mean area, scaled by the area
actually overlapped) is added to every capacity weight, so no region / GCM cell
is ever capacity-free: where there is no capacity the weighting falls back to
the plain area-weighted mean, where there is capacity the floor is negligible.

Kept free of matplotlib / cartopy / xagg so that bias_correction_qdm.py and
downscale_hourly.py can import it cheaply.
"""

import logging
import os
import sys
from pathlib import Path

# pyproj needs its own proj.db (see aggregate_cf_states.py) — before geopandas import
_proj_dir = Path(sys.prefix) / "Library" / "share" / "proj"
if _proj_dir.exists():
    os.environ["PROJ_DATA"] = str(_proj_dir)
    os.environ["PROJ_LIB"] = str(_proj_dir)

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

log = logging.getLogger(__name__)

DEFAULT_JITTER_MW = 1e-3
AEQD = "+proj=aeqd +lat_0=22 +lon_0=80 +units=m +datum=WGS84"
ALBERS = "+proj=aea +lat_1=12 +lat_2=30 +lat_0=22 +lon_0=80 +units=m +datum=WGS84"


# ══════════════════════════════════════════════════════════════════════════════
# Trackers and grids
# ══════════════════════════════════════════════════════════════════════════════

def load_tracker_projects(
    tracker_file: Path,
    country: str = "India",
    statuses: tuple = ("operating",),
) -> gpd.GeoDataFrame:
    """
    Projects of a GEM power tracker — Global Wind Power Tracker (sheets 'Data'
    + 'Below Threshold') or Global Solar Power Tracker (sheets '20 MW+' +
    '1-20 MW').  Every sheet with the project columns is read, so the same
    function serves both.  Returns points (EPSG:4326) with 'Capacity (MW)'.
    """
    need = ["Country/Area", "Status", "Capacity (MW)", "Latitude", "Longitude"]
    sheets = pd.read_excel(tracker_file, sheet_name=None)
    df = pd.concat(
        [s.assign(sheet=name) for name, s in sheets.items()
         if set(need).issubset(s.columns)],
        ignore_index=True,
    )
    df = df[(df["Country/Area"] == country) & df["Status"].isin(statuses)]
    df = df.dropna(subset=["Latitude", "Longitude", "Capacity (MW)"])
    return gpd.GeoDataFrame(
        df[["Project Name", "Phase Name", "Capacity (MW)", "Status", "sheet"]],
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )


def _edges(c: np.ndarray) -> np.ndarray:
    """Cell edges of a 1-D coordinate (midpoints, end cells mirrored)."""
    c = np.asarray(c, dtype=np.float64)
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])


def grid_bounds(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lat_b, lon_b) cell edges of a regular / rectilinear grid (lat clipped to ±90)."""
    return np.clip(_edges(lat), -90.0, 90.0), _edges(lon)


def pixel_area(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(lat, lon) spherical cell areas, relative to their mean (mean = 1)."""
    lat_b, lon_b = grid_bounds(lat, lon)
    a = np.outer(np.abs(np.diff(np.sin(np.radians(lat_b)))),
                 np.abs(np.diff(lon_b)))
    return a / a.mean()


def _nearest_index(coord: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.abs(coord[None, :] - values[:, None]).argmin(1)


def capacity_map(projects: gpd.GeoDataFrame, grid: xr.Dataset) -> xr.DataArray:
    """MW installed per pixel of *grid* (each project on its nearest pixel), (lat, lon)."""
    lat, lon = grid["lat"].values, grid["lon"].values
    ilat = _nearest_index(lat, projects.geometry.y.values)
    ilon = _nearest_index(lon, projects.geometry.x.values)
    # projects outside the grid box would pile up on its edge — drop them
    lat_b, lon_b = grid_bounds(lat, lon)
    inside = ((projects.geometry.y.values >= lat_b.min()) & (projects.geometry.y.values <= lat_b.max())
              & (projects.geometry.x.values >= lon_b.min()) & (projects.geometry.x.values <= lon_b.max()))
    w = np.zeros((len(lat), len(lon)))
    np.add.at(w, (ilat[inside], ilon[inside]), projects["Capacity (MW)"].values[inside])
    return xr.DataArray(w, dims=("lat", "lon"),
                        coords={"lat": grid["lat"], "lon": grid["lon"]}, name="capacity_mw")


def region_pixel_overlap(grid: xr.Dataset, gdf, region_col: str) -> xr.DataArray:
    """
    Area of each pixel lying inside each region, in units of the grid's mean
    pixel area (a pixel fully inside a region → its relative area, ~1).
    Returns (region, lat, lon).
    """
    lat, lon = grid["lat"].values, grid["lon"].values
    lat_b, lon_b = grid_bounds(lat, lon)
    ii, jj = np.meshgrid(np.arange(len(lat)), np.arange(len(lon)), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    pix = gpd.GeoDataFrame(
        {"ilat": ii, "ilon": jj},
        geometry=[box(lon_b[j], lat_b[i], lon_b[j + 1], lat_b[i + 1]) for i, j in zip(ii, jj)],
        crs="EPSG:4326",
    ).to_crs(ALBERS)
    mean_area = pix.geometry.area.mean()
    reg = gdf[[region_col, "geometry"]].to_crs(ALBERS)
    reg["geometry"] = reg.geometry.buffer(0)             # repair invalid polygons
    ov = gpd.overlay(pix, reg, how="intersection", keep_geom_type=True)

    regions = gdf[region_col].tolist()
    out = np.zeros((len(regions), len(lat), len(lon)))
    ireg = ov[region_col].map({r: k for k, r in enumerate(regions)}).values
    np.add.at(out, (ireg, ov["ilat"].values, ov["ilon"].values),
              ov.geometry.area.values / mean_area)
    return xr.DataArray(out, dims=("region", "lat", "lon"),
                        coords={"region": regions, "lat": grid["lat"], "lon": grid["lon"]})


# ══════════════════════════════════════════════════════════════════════════════
# State aggregation
# ══════════════════════════════════════════════════════════════════════════════

def build_capacity_weights(
    tracker_file: Path,
    grid: xr.Dataset,
    gdf,
    region_col: str,
    country: str = "India",
    statuses: tuple = ("operating",),
    max_dist_km: float = 50.0,
    jitter_mw: float = DEFAULT_JITTER_MW,
) -> tuple[xr.DataArray, pd.DataFrame]:
    """
    Installed capacity (MW) per (region, pixel) from a GEM power tracker.

    Each project with a status in *statuses* is assigned to the region that
    contains it (or the nearest region within max_dist_km, for coastal /
    border points that fall just outside the polygons) and to the nearest
    pixel of *grid*.  Capacities of projects sharing a (region, pixel) are
    summed.  The jitter floor jitter_mw × (area of the pixel inside the
    region, in mean-pixel units) is then added, so every region has weight on
    the pixels it overlaps and falls back to an area-weighted mean where it
    holds no capacity.

    Returns
    -------
    weights  : DataArray (region, lat, lon) of MW (+ jitter)
    projects : DataFrame of the retained projects with their region / pixel
    """
    pts = load_tracker_projects(tracker_file, country, statuses)
    joined = gpd.sjoin_nearest(
        pts.to_crs(AEQD), gdf[[region_col, "geometry"]].to_crs(AEQD),
        how="inner", max_distance=max_dist_km * 1000.0,
    )
    joined = joined[~joined.index.duplicated()]      # ties on shared borders
    n_drop = len(pts) - len(joined)
    if n_drop:
        log.warning("  %d %s projects farther than %g km from any region — dropped",
                    n_drop, country, max_dist_km)

    lat = grid["lat"].values
    lon = grid["lon"].values
    joined["ilat"] = _nearest_index(lat, pts.loc[joined.index].geometry.y.values)
    joined["ilon"] = _nearest_index(lon, pts.loc[joined.index].geometry.x.values)

    regions = gdf[region_col].tolist()
    w = np.zeros((len(regions), len(lat), len(lon)))
    ireg = joined[region_col].map({r: i for i, r in enumerate(regions)}).values
    np.add.at(w, (ireg, joined["ilat"].values, joined["ilon"].values),
              joined["Capacity (MW)"].values)
    weights = xr.DataArray(
        w, dims=("region", "lat", "lon"),
        coords={"region": regions, "lat": grid["lat"], "lon": grid["lon"]},
        name="capacity_mw",
    )
    log.info("  Capacity weights: %d projects, %.1f GW, %d pixels, %d/%d regions with capacity",
             len(joined), joined["Capacity (MW)"].sum() / 1e3,
             int((weights.sum("region") > 0).sum()),
             int((weights.sum(["lat", "lon"]) > 0).sum()), len(regions))
    if jitter_mw > 0:
        weights = weights + jitter_mw * region_pixel_overlap(grid, gdf, region_col)
        weights.name = "capacity_mw"
        log.info("  + capacity jitter %g MW per pixel (area-weighted fallback)", jitter_mw)
    return weights, pd.DataFrame(joined.drop(columns="geometry"))


def capacity_weighted_mean(
    cf_da: xr.DataArray, weights: xr.DataArray
) -> xr.DataArray:
    """
    Capacity-weighted CF per region, at cf_da's own time step:
        CF_r(t) = Σ_pixels MW(r, pixel) * CF(pixel, t) / Σ_pixels MW(r, pixel)
    Only pixels holding weight are touched (a few hundred), so this is cheap
    even on hourly data.  Pixels with NaN CF are excluded from both sums;
    regions with no weight (no capacity and no jitter) are NaN.
    Returns (time, region).
    """
    w  = weights.stack(pixel=("lat", "lon"))
    w  = w.isel(pixel=np.flatnonzero(w.sum("region").values > 0))
    cf = cf_da.stack(pixel=("lat", "lon")).sel(pixel=w["pixel"])
    num = xr.dot(cf.fillna(0.0), w, dims="pixel")
    den = xr.dot(cf.notnull().astype(np.float32), w, dims="pixel")
    return (num / den.where(den > 0)).transpose("time", "region").astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ERA5 → GCM re-aggregation
# ══════════════════════════════════════════════════════════════════════════════

def _grid_ds(lat, lon) -> xr.Dataset:
    lat_b, lon_b = grid_bounds(np.asarray(lat), np.asarray(lon))
    return xr.Dataset(coords={"lat": ("lat", np.asarray(lat)), "lon": ("lon", np.asarray(lon)),
                              "lat_b": ("lat_b", lat_b), "lon_b": ("lon_b", lon_b)})


def tracker_capacity_maps(grid: xr.Dataset, trackers: dict, var_tech: dict) -> dict:
    """
    {var: capacity_map on *grid*} for each var whose technology has a tracker
    file.  trackers: {"wind": path | None, "solar": path | None};
    var_tech: {var: "wind" | "solar"}.  Vars left out are area-weighted.
    """
    maps = {}
    for tech, path in trackers.items():
        if path is None:
            continue
        cap = capacity_map(load_tracker_projects(path), grid)
        log.info("  %s capacity on source grid: %.1f GW on %d pixels",
                 tech, float(cap.sum()) / 1e3, int((cap > 0).sum()))
        maps.update({v: cap for v, t in var_tech.items() if t == tech})
    return maps


class WeightedConservativeRegridder:
    """
    Conservative re-aggregation of a fine source grid (ERA5 0.25°) onto a
    coarse target grid (GCM), weighted by installed capacity:

        X_cell = Σ_j overlap(cell, j) · c_j · x_j  /  Σ_j overlap(cell, j) · c_j

    with c_j = MW_j / area_j + jitter_mw (capacity density of source pixel j
    plus the jitter floor, per mean-pixel area).  Vars without a capacity map
    get c_j = jitter_mw, i.e. a plain area-weighted conservative mean.
    Source pixels that are NaN (at the first time step) carry no weight.
    Target cells not overlapping the source domain at all take the bilinear
    (+ nearest_s2d) value instead.
    """

    def __init__(self, src_grid: xr.Dataset, dst_grid: xr.Dataset, xe,
                 capacity: dict | None = None, jitter_mw: float = DEFAULT_JITTER_MW):
        if jitter_mw <= 0:
            raise ValueError("jitter_mw must be > 0 (cells without capacity would be empty)")
        self.src = _grid_ds(src_grid["lat"].values, src_grid["lon"].values)
        self.dst = _grid_ds(dst_grid["lat"].values, dst_grid["lon"].values)
        self.conservative = xe.Regridder(self.src, self.dst, method="conservative",
                                         reuse_weights=False)
        self.bilinear = xe.Regridder(self.src, self.dst, method="bilinear",
                                     extrap_method="nearest_s2d", reuse_weights=False)
        area = pixel_area(self.src["lat"].values, self.src["lon"].values)
        self.capacity = capacity or {}
        self.jitter_mw = jitter_mw
        self._density = {
            v: xr.DataArray(cap.values / area + jitter_mw, dims=("lat", "lon"),
                            coords={"lat": self.src["lat"], "lon": self.src["lon"]})
            for v, cap in self.capacity.items()
        }
        self._uniform = xr.DataArray(np.full(area.shape, jitter_mw), dims=("lat", "lon"),
                                     coords={"lat": self.src["lat"], "lon": self.src["lon"]})
        self._den = {}   # var -> (denominator on dst grid, weight on src grid)

    def describe(self, var: str) -> str:
        how = "capacity-weighted" if var in self.capacity else "area-weighted"
        return f"{how} conservative re-aggregation (capacity jitter {self.jitter_mw:g} MW)"

    def _weights(self, var: str, da: xr.DataArray):
        if var not in self._den:
            c = self._density.get(var, self._uniform)
            first = da.isel(time=0, drop=True) if "time" in da.dims else da
            c = c.where(first.notnull(), 0.0)
            den = self.conservative(c)
            self._den[var] = (den, c)
            n_empty = int((den <= 0).sum())
            if n_empty:
                log.info("  %s: %d target cells outside source domain → bilinear/nearest",
                         var, n_empty)
        return self._den[var]

    def regrid_da(self, da: xr.DataArray, var: str | None = None) -> xr.DataArray:
        var = var or da.name
        # source coords exactly those the regridders were built on (no FP misalignment)
        da = da.assign_coords(lat=self.src["lat"].values, lon=self.src["lon"].values)
        den, c = self._weights(var, da)
        num = self.conservative((da * c).fillna(0.0))
        out = num / den.where(den > 0)
        if bool((den <= 0).any()):
            out = out.where(den > 0, self.bilinear(da))
        out = out.astype(da.dtype)
        out.attrs = dict(da.attrs)
        return out.rename(var)

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        return xr.Dataset({v: self.regrid_da(ds[v], v) for v in ds.data_vars})
