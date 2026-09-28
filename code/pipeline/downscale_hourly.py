#!/usr/bin/env python3
"""
Hourly temporal downscaling of bias-corrected GCM daily data.
STOCHASTIC VERSION — individual ERA5 profiles are stored and randomly sampled
at apply time, preserving sub-daily variability (cloud breaks, wind gusts, …).

Two-phase pipeline
------------------
fit
    Regrid ERA5 hourly NetCDF files to GCM grid with xesmf bilinear.
    Fit MiniBatchKMeans on normalised daily (rsds, tas, sfcWind) features.
    Store ALL individual per-day diurnal profiles per grid cell:
        rsds    → fraction of daily mean   (prof_frac)
        tas     → anomaly from daily mean  (prof_anom)
        sfcWind → ratio to daily mean      (prof_ratio)
    Also store cluster assignment and DOY for every ERA5 day.
    Save to library NetCDF.

apply
    Load library + BC-corrected GCM daily files.
    For each (day, cell):
        1. Normalise daily values → find nearest centroid within ±doy-window.
        2. Among ERA5 days in that cluster within ±doy-window, draw one at
           random using a deterministic seed derived from (day_index, lat_i, lon_i).
        3. Apply that individual ERA5 profile to reconstruct 24 hourly values.
    Write hourly NetCDF files ready for CF computation.

Why stochastic sampling instead of mean profiles?
    Mean profiles produce an unrealistically smooth diurnal cycle: every day
    assigned to the same cluster gets the same bell-shaped rsds curve, erasing
    cloud-break variability.  Sampling a real ERA5 day preserves the full
    sub-daily distribution, which matters for renewable energy drought metrics
    (REDs / EBDs) that aggregate over hours.

ERA5 NetCDF input
    Run convert_era5_hourly.py first to convert raw GRIB files to NetCDF
    containing {tas (K), rsds (W m-2), sfcWind (m s-1)}, zlib-compressed.

Usage
-----
# 0 – Convert GRIB → NetCDF (run once per year file):
python convert_era5_hourly.py /data/raw/era5/era5_india_*.grib \\
    --out-dir /data/proc/era5

# 1 – Fit (once per GCM grid):
python downscale_hourly.py fit \\
    --era5-nc    "/data/raw/era5/era5_india_*.nc" \\
    --gcm-grid   /data/proc/cmip6_bc/tas_CanESM5_historical_bc.nc \\
    --out-library /data/proc/era5/diurnal_library_CanESM5.nc \\
    --n-clusters  30 \\
    --env-dir     /path/to/conda/env

# 2 – Apply (per SSP):
python downscale_hourly.py apply \\
    --library  /data/proc/era5/diurnal_library_CanESM5.nc \\
    --bc-dir   /data/proc/cmip6_bc \\
    --gcm      CanESM5 --run r10i1p1f1 \\
    --ssps     ssp245 ssp585 \\
    --out-dir  /data/proc/cmip6_hourly \\
    --doy-window 30 \\
    --seed 42
"""

import os
import sys
import glob
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import MiniBatchKMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── ERA5 NetCDF loader ────────────────────────────────────────────────────────

def load_era5_hourly_nc(nc_path: Path) -> xr.Dataset:
    """
    Load one ERA5 hourly NetCDF produced by convert_era5_hourly.py.
    Expected variables: tas (K), rsds (W m-2), sfcWind (m s-1).
    """
    ds = xr.open_dataset(nc_path, chunks={"time": 24})
    rn = {}
    if "latitude" in ds.coords:
        rn["latitude"] = "lat"
    if "longitude" in ds.coords:
        rn["longitude"] = "lon"
    if rn:
        ds = ds.rename(rn)
    ds = ds.convert_calendar('standard')
    return ds.sortby("lat").sortby("lon")


# ── xesmf regridder factory ───────────────────────────────────────────────────

def make_regridder(source_ds: xr.Dataset, target_ds: xr.Dataset, xe):
    return xe.Regridder(
        source_ds, target_ds,
        method="bilinear",
        extrap_method="nearest_s2d",
        reuse_weights=False,
    )


# ── Daily aggregation from hourly ─────────────────────────────────────────────

def hourly_to_daily(ds_h: xr.Dataset) -> xr.Dataset:
    """
    Aggregate hourly → daily mean.
    rsds, tas, sfcWind: daily mean (W m-2, K, m s-1).
    """
    rsds_day = ds_h["rsds"].resample(time="1D").mean()
    tas_day  = ds_h["tas"].resample(time="1D").mean()
    wind_day = ds_h["sfcWind"].resample(time="1D").mean()
    return xr.Dataset({"rsds": rsds_day, "tas": tas_day, "sfcWind": wind_day})


def hourly_profiles(ds_h: xr.Dataset, ds_d: xr.Dataset):
    """
    Compute per-day, per-cell, 24-h decomposition profiles.

    Returns arrays shaped (n_days, 24, n_lat, n_lon):
        frac   – rsds ratio to daily mean    (mean 1 per day/cell, 0 at night)
        anom   – tas anomaly from daily mean  (mean 0 per day/cell)
        ratio  – sfcWind ratio to daily mean  (mean 1 per day/cell)
    """
    n_lat = ds_h.dims["lat"]
    n_lon = ds_h.dims["lon"]
    n_days = ds_d.dims["time"]

    rsds_h = ds_h["rsds"].values    # (n_days*24, lat, lon)
    tas_h  = ds_h["tas"].values
    wind_h = ds_h["sfcWind"].values

    rsds_d = ds_d["rsds"].values    # (n_days, lat, lon)
    tas_d  = ds_d["tas"].values
    wind_d = ds_d["sfcWind"].values

    rsds_h3 = rsds_h.reshape(n_days, 24, n_lat, n_lon)
    tas_h3  = tas_h.reshape(n_days, 24, n_lat, n_lon)
    wind_h3 = wind_h.reshape(n_days, 24, n_lat, n_lon)

    denom_rsds = rsds_d[:, np.newaxis, :, :]
    frac = np.where(denom_rsds > 0, rsds_h3 / denom_rsds, 0.0)

    anom = tas_h3 - tas_d[:, np.newaxis, :, :]

    denom_wind = wind_d[:, np.newaxis, :, :]
    ratio = np.where(denom_wind > 0, wind_h3 / denom_wind, 1.0)

    return frac.astype(np.float32), anom.astype(np.float32), ratio.astype(np.float32)


# ── Calendar-window cluster filter ───────────────────────────────────────────

def circular_doy_distance(doys_a: np.ndarray, doy_b: int) -> np.ndarray:
    """Circular distance (days) between an array of DOYs and a scalar DOY."""
    return np.abs(((doys_a - doy_b + 182) % 365) - 182)


def get_valid_clusters(doy: int, cluster_doys: np.ndarray, window: int) -> np.ndarray:
    """
    Return indices of clusters whose circular-mean DOY falls within ±window
    days of the target DOY.
    """
    diff = circular_doy_distance(cluster_doys, doy)
    valid = np.where(diff <= window)[0]
    if len(valid) == 0:
        log.warning("DOY %d: no cluster within window=%d — using all clusters", doy, window)
        valid = np.arange(len(cluster_doys))
    return valid


# ── Feature matrix builder ───────────────────────────────────────────────────

def build_features(ds_d: xr.Dataset, stats: dict | None = None):
    """
    Flatten daily dataset → (n_days * n_lat * n_lon, 3) feature matrix.
    Normalise by per-cell mean/std computed from ERA5 (or supplied as stats).

    Returns (X_norm, stats) where stats = {"mean": ..., "std": ...}
    each shaped (n_lat, n_lon, 3).
    """
    rsds_d = ds_d["rsds"].values
    tas_d  = ds_d["tas"].values
    wind_d = ds_d["sfcWind"].values

    X = np.stack([rsds_d, tas_d, wind_d], axis=-1)  # (n_days, lat, lon, 3)
    n_d, n_lat, n_lon, n_v = X.shape

    if stats is None:
        mean = X.mean(axis=0)
        std  = X.std(axis=0) + 1e-8
        stats = {"mean": mean, "std": std}

    X_norm = (X - stats["mean"][np.newaxis]) / stats["std"][np.newaxis]
    X_flat = X_norm.reshape(-1, n_v)
    return X_flat, stats


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 – FIT
# ══════════════════════════════════════════════════════════════════════════════

def cmd_fit(args):
    env_dir = Path(args.env_dir) if args.env_dir else Path(sys.prefix)
    mk = env_dir / "Library" / "lib" / "esmf.mk"
    if not mk.exists():
        mk = env_dir / "lib" / "esmf.mk"
    os.environ["ESMFMKFILE"] = str(mk)
    log.info("ESMFMKFILE = %s  (exists: %s)", mk, mk.exists())
    import xesmf as xe

    nc_files = sorted(glob.glob(args.era5_nc))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files matched: {args.era5_nc}")
    log.info("Found %d NetCDF file(s)", len(nc_files))

    # Load target grid from one BC file
    gcm_grid_ds = xr.open_dataset(args.gcm_grid).isel(time=0).drop_vars("time", errors="ignore")
    gcm_grid_ds = gcm_grid_ds[list(gcm_grid_ds.data_vars)[:1]]
    lat_gcm = gcm_grid_ds["lat"].values
    lon_gcm = gcm_grid_ds["lon"].values
    n_lat, n_lon = len(lat_gcm), len(lon_gcm)
    log.info("GCM grid: %d lat × %d lon", n_lat, n_lon)

    # ── Pass 1: collect daily features for K-means ────────────────────────────
    log.info("=== Pass 1: build feature matrix ===")
    regridder = None
    all_daily_list = []

    # Per-pixel max of the raw ERA5 hourly values — used in `apply` as a physical
    # clip ceiling, since a stochastic profile calibrated on a low daily-mean ERA5
    # day can otherwise produce hourly values far above anything ever observed.
    rsds_pixel_max = np.full((n_lat, n_lon), -np.inf, dtype=np.float32)
    wind_pixel_max = np.full((n_lat, n_lon), -np.inf, dtype=np.float32)

    for nc_path in nc_files:
        log.info("  Loading %s …", Path(nc_path).name)
        ds_h_era5 = load_era5_hourly_nc(Path(nc_path))

        if regridder is None:
            src_grid = ds_h_era5.isel(time=0).drop_vars("time", errors="ignore")
            regridder = make_regridder(src_grid, gcm_grid_ds, xe)
            log.info("  Regridder built")

        ds_h_rg = regridder(ds_h_era5)
        ds_h_rg = ds_h_rg.assign_coords(lat=lat_gcm, lon=lon_gcm)

        rsds_pixel_max = np.maximum(rsds_pixel_max, ds_h_rg["rsds"].max("time").values)
        wind_pixel_max = np.maximum(wind_pixel_max, ds_h_rg["sfcWind"].max("time").values)

        ds_d = hourly_to_daily(ds_h_rg)
        all_daily_list.append(ds_d)
        del ds_h_era5, ds_h_rg

    ds_all_daily = xr.concat(all_daily_list, dim="time")
    del all_daily_list

    n_days_total = ds_all_daily.dims["time"]
    log.info("Total ERA5 days: %d", n_days_total)

    X_flat, era5_stats = build_features(ds_all_daily)
    log.info("Feature matrix: %s", X_flat.shape)

    # ── Fit MiniBatchKMeans ────────────────────────────────────────────────────
    log.info("=== Fitting MiniBatchKMeans (k=%d) ===", args.n_clusters)
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=42,
        batch_size=min(10_000, len(X_flat)),
        n_init=10,
    )
    labels_flat = kmeans.fit_predict(X_flat)     # (n_days*lat*lon,)
    labels = labels_flat.reshape(n_days_total, n_lat, n_lon)
    log.info("Inertia: %.3e", kmeans.inertia_)
    del X_flat

    # ── Compute circular-mean DOY per cluster ─────────────────────────────────
    log.info("=== Computing circular-mean DOY per cluster ===")
    K = args.n_clusters
    times_all = pd.DatetimeIndex(ds_all_daily.time.values)
    doys_all  = np.array([t.dayofyear for t in times_all])
    doys_flat = np.repeat(doys_all, n_lat * n_lon)

    cluster_doys = np.zeros(K, dtype=np.float64)
    for k in range(K):
        mask_k = labels_flat == k
        if mask_k.any():
            angles = doys_flat[mask_k] * (2 * np.pi / 365)
            circular_mean = (
                np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())) % 360
            ) * 365 / 360
            cluster_doys[k] = max(1.0, circular_mean)
        else:
            cluster_doys[k] = 183.0
    log.info("Cluster DOY range: %.1f – %.1f", cluster_doys.min(), cluster_doys.max())

    # ── Pass 2: store ALL individual profiles ─────────────────────────────────
    # Shape: (n_days_total, 24, n_lat, n_lon) for each of frac/anom/ratio
    # Also store per-ERA5-day cluster assignment and DOY.
    #
    # NOTE: cluster labels are computed per (day, cell) in labels[d, i, j].
    # For the stochastic draw we need a single cluster label per day (not per
    # cell) to index into the profile library efficiently.  We use the
    # spatial mode (most frequent cluster across cells) as the day-level label.
    # The per-cell assignment is still used in apply for the distance search.
    log.info("=== Pass 2: collect individual profiles ===")

    # Per-day spatial-mode cluster label (used to index profile pool at apply)
    from scipy import stats as scipy_stats
    day_cluster = np.array([
        scipy_stats.mode(labels[d].ravel(), keepdims=False).mode
        for d in range(n_days_total)
    ], dtype=np.int32)   # (n_days_total,)

    # DOY for every ERA5 day
    day_doy = doys_all.astype(np.int32)   # (n_days_total,)

    all_frac_list  = []
    all_anom_list  = []
    all_ratio_list = []

    for nc_path in nc_files:
        log.info("  Profiles from %s …", Path(nc_path).name)
        ds_h_era5 = load_era5_hourly_nc(Path(nc_path))
        ds_h_rg   = regridder(ds_h_era5)
        ds_h_rg   = ds_h_rg.assign_coords(lat=lat_gcm, lon=lon_gcm)
        ds_d_year = hourly_to_daily(ds_h_rg)

        frac, anom, ratio = hourly_profiles(ds_h_rg, ds_d_year)
        # frac/anom/ratio: (n_days_year, 24, lat, lon) float32
        all_frac_list.append(frac)
        all_anom_list.append(anom)
        all_ratio_list.append(ratio)

        del ds_h_era5, ds_h_rg, ds_d_year

    # Concatenate along day axis → (n_days_total, 24, lat, lon)
    all_frac  = np.concatenate(all_frac_list,  axis=0)
    all_anom  = np.concatenate(all_anom_list,  axis=0)
    all_ratio = np.concatenate(all_ratio_list, axis=0)
    del all_frac_list, all_anom_list, all_ratio_list

    log.info("Profile arrays shape: %s", all_frac.shape)

    # ── Save library ───────────────────────────────────────────────────────────
    log.info("=== Saving library ===")
    hour_coord = np.arange(24)
    k_coord    = np.arange(K)
    day_coord  = np.arange(n_days_total)

    ds_lib = xr.Dataset({
        # Individual ERA5 diurnal profiles — (era5_day, hour, lat, lon)
        "prof_frac":  xr.DataArray(
            all_frac,
            dims=["era5_day", "hour", "lat", "lon"],
            attrs={"description": "rsds ratio to daily mean (0 at night, peak ~2 at noon)"},
        ),
        "prof_anom":  xr.DataArray(
            all_anom,
            dims=["era5_day", "hour", "lat", "lon"],
            attrs={"description": "tas anomaly from daily mean (K), mean=0 per day"},
        ),
        "prof_ratio": xr.DataArray(
            all_ratio,
            dims=["era5_day", "hour", "lat", "lon"],
            attrs={"description": "sfcWind ratio to daily mean, mean=1 per day"},
        ),
        # Per-ERA5-day metadata
        "day_cluster": xr.DataArray(
            day_cluster,
            dims=["era5_day"],
            attrs={"description": "Spatial-mode cluster label for each ERA5 day"},
        ),
        "day_doy": xr.DataArray(
            day_doy,
            dims=["era5_day"],
            attrs={"description": "Day-of-year (1–365/366) for each ERA5 day"},
        ),
        # K-means centroids
        "centroids": xr.DataArray(
            kmeans.cluster_centers_,
            dims=["cluster", "feature"],
            attrs={"features": "rsds, tas, sfcWind (normalised)"},
        ),
        # Circular-mean DOY per cluster
        "cluster_doy": xr.DataArray(
            cluster_doys,
            dims=["cluster"],
            attrs={"description": "Circular-mean DOY of ERA5 days assigned to cluster"},
        ),
        # Per-cell normalisation stats
        "feat_mean": xr.DataArray(era5_stats["mean"], dims=["lat", "lon", "feature"]),
        "feat_std":  xr.DataArray(era5_stats["std"],  dims=["lat", "lon", "feature"]),
        # Per-cell max of the raw ERA5 hourly values — physical clip ceiling for apply
        "rsds_max": xr.DataArray(
            rsds_pixel_max,
            dims=["lat", "lon"],
            attrs={"description": "Max hourly rsds observed in ERA5 (W m-2), per pixel"},
        ),
        "sfcWind_max": xr.DataArray(
            wind_pixel_max,
            dims=["lat", "lon"],
            attrs={"description": "Max hourly sfcWind observed in ERA5 (m s-1), per pixel"},
        ),
    }, coords={
        "era5_day": day_coord,
        "hour":     hour_coord,
        "cluster":  k_coord,
        "lat":      lat_gcm,
        "lon":      lon_gcm,
        "feature":  ["rsds", "tas", "sfcWind"],
    })
    ds_lib.attrs = {
        "description": "ERA5 diurnal downscaling library — stochastic individual profiles",
        "n_clusters":  K,
        "gcm_grid":    str(args.gcm_grid),
        "era5_nc":     str(args.era5_nc),
    }
    Path(args.out_library).parent.mkdir(parents=True, exist_ok=True)

    # Chunking: one ERA5 day at a time along era5_day for efficient random access
    encoding = {
        "prof_frac":  {"zlib": True, "complevel": 4, "chunksizes": (1, 24, n_lat, n_lon)},
        "prof_anom":  {"zlib": True, "complevel": 4, "chunksizes": (1, 24, n_lat, n_lon)},
        "prof_ratio": {"zlib": True, "complevel": 4, "chunksizes": (1, 24, n_lat, n_lon)},
    }
    ds_lib.to_netcdf(args.out_library, encoding=encoding)
    log.info("Library saved → %s", args.out_library)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 – APPLY  (stochastic)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_apply(args):
    log.info("Loading library …")
    lib = xr.open_dataset(args.library)

    centroids    = lib["centroids"].values       # (K, 3)
    feat_mean    = lib["feat_mean"].values        # (lat, lon, 3)
    feat_std     = lib["feat_std"].values         # (lat, lon, 3)
    cluster_doys = lib["cluster_doy"].values      # (K,)
    day_cluster  = lib["day_cluster"].values      # (N_era5,)  int32
    day_doy      = lib["day_doy"].values          # (N_era5,)  int32
    rsds_max     = lib["rsds_max"].values          # (lat, lon) — ERA5-observed ceiling
    wind_max     = lib["sfcWind_max"].values       # (lat, lon) — ERA5-observed ceiling

    # Load all profiles into memory (small for 365 days × 110 cells)
    prof_frac  = lib["prof_frac"].values          # (N_era5, 24, lat, lon)
    prof_anom  = lib["prof_anom"].values
    prof_ratio = lib["prof_ratio"].values

    lat_gcm = lib["lat"].values
    lon_gcm = lib["lon"].values
    K       = len(lib["cluster"])
    N_era5  = len(lib["era5_day"])
    n_lat, n_lon = len(lat_gcm), len(lon_gcm)

    log.info("Library: %d clusters, %d ERA5 days, grid %d×%d",
             K, N_era5, n_lat, n_lon)
    log.info("Calendar window: ±%d days | seed: %d", args.doy_window, args.seed)

    # Pre-build per-cluster index arrays for fast candidate lookup
    # cluster_members[k] = sorted array of ERA5-day indices assigned to cluster k
    cluster_members = {
        k: np.where(day_cluster == k)[0] for k in range(K)
    }

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for ssp in args.ssps:
        out_path = Path(args.out_dir) / f"{args.gcm}_{ssp}_hourly.nc"
        if out_path.exists() and not args.force:
            log.info("%s: already exists — skipping", out_path.name)
            continue

        log.info("=== %s %s ===", args.gcm, ssp)

        def load_bc(vname):
            p = Path(args.bc_dir) / f"{vname}_{args.gcm}_{ssp}_bc.nc"
            da = xr.open_dataset(p)[vname]
            da = da.assign_coords(lat=lat_gcm, lon=lon_gcm)
            da = da.convert_calendar('standard')
            da = da.transpose("time", "lat", "lon")  
            return da.load()

        rsds_d = load_bc("rsds")
        tas_d  = load_bc("tas")
        wind_d = load_bc("sfcWind")

        times_d = pd.DatetimeIndex(rsds_d.time.values)
        n_days  = len(times_d)
        log.info("BC days loaded: %d", n_days)

        # Normalise GCM daily features
        X_gcm  = np.stack([rsds_d.values, tas_d.values, wind_d.values], axis=-1)
        X_norm = (X_gcm - feat_mean[np.newaxis]) / feat_std[np.newaxis]
        # (n_days, lat, lon, 3)

        # ── Cluster assignment with calendar-window filter ────────────────────
        log.info("Assigning clusters …")
        gcm_cluster = np.empty((n_days, n_lat, n_lon), dtype=np.int32)
        for d in range(n_days):
            doy   = times_d[d].dayofyear
            valid = get_valid_clusters(doy, cluster_doys, args.doy_window)
            x_day = X_norm[d].reshape(-1, 3)                          # (lat*lon, 3)
            diff  = x_day[:, np.newaxis, :] - centroids[valid][np.newaxis]
            dist2 = (diff ** 2).sum(axis=-1)                          # (lat*lon, |valid|)
            gcm_cluster[d] = valid[dist2.argmin(axis=-1)].reshape(n_lat, n_lon)
            if d % 365 == 0:
                log.info("  Day %d/%d  DOY=%d  %d valid clusters",
                         d, n_days, doy, len(valid))

        # ── Stochastic profile draw ───────────────────────────────────────────
        # For each (day d, lat i, lon j):
        #   1. candidate pool = ERA5 days with same cluster AND |DOY - target_DOY| ≤ window
        #   2. draw one index using seed derived from (d, i, j, global_seed)
        #   3. use that ERA5 day's profile for cell (i, j)
        #
        # Vectorised over cells for each day to keep the loop manageable.

        log.info("Stochastic profile sampling and reconstruction …")
        rsds_hourly = np.empty((n_days, 24, n_lat, n_lon), dtype=np.float32)
        tas_hourly  = np.empty((n_days, 24, n_lat, n_lon), dtype=np.float32)
        wind_hourly = np.empty((n_days, 24, n_lat, n_lon), dtype=np.float32)

        rsds_v = rsds_d.values   # (n_days, lat, lon)
        tas_v  = tas_d.values
        wind_v = wind_d.values

        for d in range(n_days):
            doy = times_d[d].dayofyear

            for i in range(n_lat):
                for j in range(n_lon):
                    k = gcm_cluster[d, i, j]

                    # Candidate ERA5 days: same cluster, DOY within window
                    cands = cluster_members[k]
                    if len(cands) > 0:
                        doy_dist = circular_doy_distance(day_doy[cands], doy)
                        cands = cands[doy_dist <= args.doy_window]

                    # Fallback 1: same cluster, ignore DOY window
                    if len(cands) == 0:
                        cands = cluster_members[k]

                    # Fallback 2: any ERA5 day within DOY window (any cluster)
                    if len(cands) == 0:
                        doy_dist_all = circular_doy_distance(day_doy, doy)
                        cands = np.where(doy_dist_all <= args.doy_window)[0]

                    # Fallback 3: all ERA5 days (should never be needed)
                    if len(cands) == 0:
                        cands = np.arange(N_era5)

                    # Deterministic seed: mix global seed + day + cell indices
                    rng = np.random.default_rng(
                        seed=args.seed ^ (d * 100003 + i * 997 + j)
                    )
                    chosen = rng.choice(cands)

                    rsds_hourly[d, :, i, j] = (
                        rsds_v[d, i, j] * prof_frac[chosen, :, i, j]
                    )
                    tas_hourly[d, :, i, j] = (
                        tas_v[d, i, j] + prof_anom[chosen, :, i, j]
                    )
                    wind_hourly[d, :, i, j] = (
                        wind_v[d, i, j] * prof_ratio[chosen, :, i, j]
                    )

            if d % 365 == 0:
                log.info("  Reconstructed day %d / %d", d, n_days)

        # Clip physical bounds — ceiling = per-pixel max observed in ERA5 (library).
        # A profile calibrated on a low daily-mean ERA5 day can have a very high
        # frac/ratio; applied to a higher-mean GCM day that can exceed anything
        # ever physically observed at that pixel without this ceiling.
        rsds_hourly = np.clip(rsds_hourly, 0.0, rsds_max[np.newaxis, np.newaxis, :, :])
        wind_hourly = np.clip(wind_hourly, 0.0, wind_max[np.newaxis, np.newaxis, :, :])

        # Build output time index
        time_hourly     = pd.date_range(start=times_d[0], periods=n_days * 24, freq="h")
        rsds_hourly_2d  = rsds_hourly.reshape(n_days * 24, n_lat, n_lon)
        tas_hourly_2d   = tas_hourly.reshape(n_days * 24, n_lat, n_lon)
        wind_hourly_2d  = wind_hourly.reshape(n_days * 24, n_lat, n_lon)

        ds_out = xr.Dataset({
            "rsds": xr.DataArray(
                rsds_hourly_2d,
                dims=["time", "lat", "lon"],
                coords={"time": time_hourly, "lat": lat_gcm, "lon": lon_gcm},
                attrs={"units": "W m-2",
                       "long_name": "Surface downwelling shortwave radiation"},
            ),
            "tas": xr.DataArray(
                tas_hourly_2d,
                dims=["time", "lat", "lon"],
                coords={"time": time_hourly, "lat": lat_gcm, "lon": lon_gcm},
                attrs={"units": "K", "long_name": "Near-surface air temperature"},
            ),
            "sfcWind": xr.DataArray(
                wind_hourly_2d,
                dims=["time", "lat", "lon"],
                coords={"time": time_hourly, "lat": lat_gcm, "lon": lon_gcm},
                attrs={"units": "m s-1", "long_name": "Near-surface wind speed"},
            ),
        })
        ds_out.attrs = {
            "description": (
                f"Hourly downscaled BC GCM data — {args.gcm} {args.run} {ssp}"
            ),
            "method": (
                "Stochastic K-means diurnal profile downscaling from ERA5 hourly. "
                f"Calendar window ±{args.doy_window} days. Seed {args.seed}."
            ),
            "library":    str(args.library),
            "gcm":        args.gcm,
            "run":        args.run,
            "ssp":        ssp,
            "doy_window": args.doy_window,
            "seed":       args.seed,
        }

        encoding = {
            v: {"zlib": True, "complevel": 4,
                "chunksizes": (24, n_lat, n_lon)}
            for v in ds_out.data_vars
        }
        ds_out.to_netcdf(out_path, encoding=encoding)
        log.info("→ %s", out_path.name)
        del rsds_d, tas_d, wind_d, rsds_hourly, tas_hourly, wind_hourly, ds_out

    log.info("All SSPs done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Hourly temporal downscaling of BC GCM daily data (stochastic)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="Train K-means diurnal library from ERA5 hourly")
    fit.add_argument("--era5-nc",     required=True,
                     help="Glob pattern for ERA5 hourly NetCDF files")
    fit.add_argument("--gcm-grid",    required=True, type=Path,
                     help="Any BC output NetCDF (used only for lat/lon grid)")
    fit.add_argument("--out-library", required=True,
                     help="Output library NetCDF path")
    fit.add_argument("--n-clusters",  type=int, default=30,
                     help="Number of K-means clusters (default: 30)")
    fit.add_argument("--env-dir",     type=Path, default=None,
                     help="Conda env root for ESMFMKFILE (default: sys.prefix)")

    app = sub.add_parser("apply", help="Apply diurnal library to BC GCM daily data")
    app.add_argument("--library",    required=True,
                     help="Library NetCDF produced by 'fit'")
    app.add_argument("--bc-dir",     required=True, type=Path,
                     help="Directory with bias-corrected daily NetCDF files")
    app.add_argument("--gcm",        default="CanESM5")
    app.add_argument("--run",        default="r10i1p1f1")
    app.add_argument("--ssps",       nargs="+", default=["ssp245", "ssp585"])
    app.add_argument("--out-dir",    required=True, type=Path,
                     help="Output directory for hourly NetCDF files")
    app.add_argument("--doy-window", type=int, default=30,
                     help="Calendar half-window in days (default: 30)")
    app.add_argument("--seed",       type=int, default=42,
                     help="Base seed for reproducible stochastic sampling (default: 42)")
    app.add_argument("--force",      action="store_true",
                     help="Recompute and overwrite hourly files that already exist")

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "fit":
        cmd_fit(args)
    else:
        cmd_apply(args)


if __name__ == "__main__":
    main()