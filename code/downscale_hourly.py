#!/usr/bin/env python3
"""
Hourly temporal downscaling of bias-corrected GCM daily data.

Two-phase pipeline
------------------
fit
    Regrid ERA5 hourly NetCDF files to GCM grid with xesmf bilinear.
    Fit MiniBatchKMeans on normalised daily (rsds, tas, sfcWind) features.
    Compute per-cluster diurnal profiles per grid cell:
        rsds    → fraction of daily total   (sums to 1)
        tas     → anomaly from daily mean   (mean  0)
        sfcWind → ratio to daily mean       (mean  1)
    Also store the circular-mean day-of-year of each cluster for calendar
    filtering.  Save centroid matrix, normalisation stats, cluster DOYs and
    profiles as library NetCDF.

apply
    Load library + BC-corrected GCM daily files.
    For each (day, cell) normalise daily values → find nearest centroid
    **within a ±doy-window calendar window** → reconstruct 24 hourly values
    using the cell's cluster profile.
    Write hourly NetCDF files ready for CF computation.

Calendar window rationale
    Restricting centroid search to clusters whose median DOY is within ±N
    days of the target day prevents a January day from inheriting a July
    diurnal profile even when their daily rsds/tas/wind values are similar
    (common in tropical regions or for low-seasonality variables such as
    wind).  Distance is computed circularly so the window wraps correctly
    around 1 Jan / 31 Dec.

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
    --doy-window 30
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
    Aggregate hourly → daily.
    rsds: sum of 24 hourly W m-2 values (→ daily mean-equivalent total)
    tas, sfcWind: daily mean
    """
    rsds_day  = ds_h["rsds"].resample(time="1D").sum()
    tas_day   = ds_h["tas"].resample(time="1D").mean()
    wind_day  = ds_h["sfcWind"].resample(time="1D").mean()
    return xr.Dataset({"rsds": rsds_day, "tas": tas_day, "sfcWind": wind_day})


def hourly_profiles(ds_h: xr.Dataset, ds_d: xr.Dataset):
    """
    Compute per-day, per-cell, 24-h decomposition profiles.

    Returns arrays shaped (n_days, 24, n_lat, n_lon):
        frac   – rsds fraction of daily sum      (sums to 1 per day/cell)
        anom   – tas anomaly from daily mean      (sum 0 per day/cell)
        ratio  – sfcWind ratio to daily mean      (mean 1 per day/cell)
    """
    n_lat = ds_h.dims["lat"]
    n_lon = ds_h.dims["lon"]
    times_h = pd.DatetimeIndex(ds_h.time.values)
    dates_d = pd.DatetimeIndex(ds_d.time.values)
    n_days = len(dates_d)

    rsds_h   = ds_h["rsds"].values    # (n_days*24, lat, lon)
    tas_h    = ds_h["tas"].values
    wind_h   = ds_h["sfcWind"].values

    rsds_d   = ds_d["rsds"].values    # (n_days, lat, lon)
    tas_d    = ds_d["tas"].values
    wind_d   = ds_d["sfcWind"].values

    rsds_h3  = rsds_h.reshape(n_days, 24, n_lat, n_lon)
    tas_h3   = tas_h.reshape(n_days, 24, n_lat, n_lon)
    wind_h3  = wind_h.reshape(n_days, 24, n_lat, n_lon)

    # rsds fraction: avoid division by zero (night-only days stay 0)
    denom_rsds = rsds_d[:, np.newaxis, :, :]
    frac = np.where(denom_rsds > 0, rsds_h3 / denom_rsds, 0.0)

    # tas anomaly
    anom = tas_h3 - tas_d[:, np.newaxis, :, :]

    # sfcWind ratio: avoid division by zero (calm days stay 1)
    denom_wind = wind_d[:, np.newaxis, :, :]
    ratio = np.where(denom_wind > 0, wind_h3 / denom_wind, 1.0)

    return frac, anom, ratio


# ── Calendar-window cluster filter ───────────────────────────────────────────

def get_valid_clusters(doy: int, cluster_doys: np.ndarray, window: int) -> np.ndarray:
    """
    Return indices of clusters whose circular-mean DOY falls within ±window
    days of the target DOY.  Distance wraps correctly around 1 Jan / 31 Dec.
    Falls back to all clusters if none qualify (should not happen for window ≥ 15).
    """
    diff = np.abs(((cluster_doys - doy + 182) % 365) - 182)
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
    each shaped (3, n_lat, n_lon).
    """
    rsds_d = ds_d["rsds"].values   # (n_days, lat, lon)
    tas_d  = ds_d["tas"].values
    wind_d = ds_d["sfcWind"].values

    X = np.stack([rsds_d, tas_d, wind_d], axis=-1)  # (n_days, lat, lon, 3)
    n_d, n_lat, n_lon, n_v = X.shape

    if stats is None:
        mean = X.mean(axis=0)            # (lat, lon, 3)
        std  = X.std(axis=0) + 1e-8
        stats = {"mean": mean, "std": std}

    X_norm = (X - stats["mean"][np.newaxis]) / stats["std"][np.newaxis]
    X_flat = X_norm.reshape(-1, n_v)    # (n_days*lat*lon, 3)
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

    # Load target grid from one BC file (drop time so it becomes a 2-D template)
    gcm_grid_ds = xr.open_dataset(args.gcm_grid).isel(time=0).drop_vars("time", errors="ignore")
    # keep only lat/lon dimensions for regridder
    gcm_grid_ds = gcm_grid_ds[list(gcm_grid_ds.data_vars)[:1]]
    lat_gcm = gcm_grid_ds["lat"].values
    lon_gcm = gcm_grid_ds["lon"].values
    n_lat, n_lon = len(lat_gcm), len(lon_gcm)
    log.info("GCM grid: %d lat × %d lon", n_lat, n_lon)

    # ── Pass 1: collect daily features for K-means ────────────────────────────
    log.info("=== Pass 1: build feature matrix ===")
    regridder = None
    all_daily_list = []

    for nc_path in nc_files:
        log.info("  Loading %s …", Path(nc_path).name)
        ds_h_era5 = load_era5_hourly_nc(Path(nc_path))

        if regridder is None:
            src_grid = ds_h_era5.isel(time=0).drop_vars("time", errors="ignore")
            regridder = make_regridder(src_grid, gcm_grid_ds, xe)
            log.info("  Regridder built")

        ds_h_rg = regridder(ds_h_era5)
        ds_h_rg = ds_h_rg.assign_coords(lat=lat_gcm, lon=lon_gcm)
        ds_d = hourly_to_daily(ds_h_rg)
        all_daily_list.append(ds_d)
        del ds_h_era5, ds_h_rg

    ds_all_daily = xr.concat(all_daily_list, dim="time")
    del all_daily_list

    log.info("Total ERA5 days: %d", ds_all_daily.dims["time"])

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
    n_days_total = ds_all_daily.dims["time"]
    labels = labels_flat.reshape(n_days_total, n_lat, n_lon)
    log.info("Inertia: %.3e", kmeans.inertia_)
    del X_flat

    # ── Compute circular-mean DOY per cluster ─────────────────────────────────
    log.info("=== Computing circular-mean DOY per cluster ===")
    K = args.n_clusters
    times_all = pd.DatetimeIndex(ds_all_daily.time.values)
    doys_all  = np.array([t.dayofyear for t in times_all])   # (n_days,)
    # repeat each DOY for every grid cell so it aligns with labels_flat
    doys_flat = np.repeat(doys_all, n_lat * n_lon)            # (n_days*lat*lon,)

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
            cluster_doys[k] = 183.0   # fallback: mid-year
    log.info("Cluster DOY range: %.1f – %.1f", cluster_doys.min(), cluster_doys.max())

    # ── Pass 2: compute per-cluster per-cell profiles ─────────────────────────
    log.info("=== Pass 2: accumulate per-cluster profiles ===")
    # accumulators: (n_clusters, 24, n_lat, n_lon)
    sum_frac  = np.zeros((K, 24, n_lat, n_lon), dtype=np.float64)
    sum_anom  = np.zeros((K, 24, n_lat, n_lon), dtype=np.float64)
    sum_ratio = np.zeros((K, 24, n_lat, n_lon), dtype=np.float64)
    count_k   = np.zeros((K, n_lat, n_lon), dtype=np.float64)

    day_offset = 0
    for nc_path in nc_files:
        log.info("  Profiles from %s …", Path(nc_path).name)
        ds_h_era5 = load_era5_hourly_nc(Path(nc_path))
        ds_h_rg   = regridder(ds_h_era5)
        ds_h_rg   = ds_h_rg.assign_coords(lat=lat_gcm, lon=lon_gcm)
        ds_d_year = hourly_to_daily(ds_h_rg)
        n_days_year = ds_d_year.dims["time"]

        frac, anom, ratio = hourly_profiles(ds_h_rg, ds_d_year)
        lbl_year = labels[day_offset: day_offset + n_days_year]  # (n_d_yr, lat, lon)

        for k in range(K):
            mask = (lbl_year == k)  # (n_d_yr, lat, lon)
            if not mask.any():
                continue
            # mask broadcast over hours: (n_d_yr, 24, lat, lon)
            mask24 = mask[:, np.newaxis, :, :]
            sum_frac[k]  += (frac  * mask24).sum(axis=0)
            sum_anom[k]  += (anom  * mask24).sum(axis=0)
            sum_ratio[k] += (ratio * mask24).sum(axis=0)
            count_k[k]   += mask.sum(axis=0).astype(np.float64)

        day_offset += n_days_year
        del ds_h_era5, ds_h_rg, ds_d_year, frac, anom, ratio, lbl_year

    # Normalise accumulators
    cnt = np.maximum(count_k[:, np.newaxis, :, :], 1)
    prof_frac  = sum_frac  / cnt   # (K, 24, lat, lon)
    prof_anom  = sum_anom  / cnt
    prof_ratio = sum_ratio / cnt

    # ── Save library ───────────────────────────────────────────────────────────
    log.info("=== Saving library ===")
    hour_coord = np.arange(24)
    k_coord    = np.arange(K)

    ds_lib = xr.Dataset({
        # diurnal profiles
        "prof_frac":  xr.DataArray(prof_frac,  dims=["cluster", "hour", "lat", "lon"]),
        "prof_anom":  xr.DataArray(prof_anom,  dims=["cluster", "hour", "lat", "lon"]),
        "prof_ratio": xr.DataArray(prof_ratio, dims=["cluster", "hour", "lat", "lon"]),
        # K-means centroids (normalised feature space)
        "centroids":  xr.DataArray(
            kmeans.cluster_centers_,
            dims=["cluster", "feature"],
            attrs={"features": "rsds, tas, sfcWind (normalised)"},
        ),
        # circular-mean DOY per cluster (used for calendar-window filtering)
        "cluster_doy": xr.DataArray(
            cluster_doys,
            dims=["cluster"],
            attrs={"description": "Circular-mean day-of-year of ERA5 days assigned to cluster"},
        ),
        # per-cell normalisation stats
        "feat_mean": xr.DataArray(era5_stats["mean"], dims=["lat", "lon", "feature"]),
        "feat_std":  xr.DataArray(era5_stats["std"],  dims=["lat", "lon", "feature"]),
    }, coords={
        "cluster": k_coord,
        "hour": hour_coord,
        "lat": lat_gcm,
        "lon": lon_gcm,
        "feature": ["rsds", "tas", "sfcWind"],
    })
    ds_lib.attrs = {
        "description": "ERA5 diurnal downscaling library (K-means cluster profiles)",
        "n_clusters":  K,
        "gcm_grid":    args.gcm_grid,
        "era5_nc":     args.era5_nc,
    }
    Path(args.out_library).parent.mkdir(parents=True, exist_ok=True)
    ds_lib.to_netcdf(args.out_library)
    log.info("Library saved → %s", args.out_library)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 – APPLY
# ══════════════════════════════════════════════════════════════════════════════

def cmd_apply(args):
    log.info("Loading library …")
    lib = xr.open_dataset(args.library)
    centroids    = lib["centroids"].values        # (K, 3)
    feat_mean    = lib["feat_mean"].values        # (lat, lon, 3)
    feat_std     = lib["feat_std"].values         # (lat, lon, 3)
    prof_frac    = lib["prof_frac"].values        # (K, 24, lat, lon)
    prof_anom    = lib["prof_anom"].values        # (K, 24, lat, lon)
    prof_ratio   = lib["prof_ratio"].values       # (K, 24, lat, lon)
    cluster_doys = lib["cluster_doy"].values      # (K,)
    lat_gcm = lib["lat"].values
    lon_gcm = lib["lon"].values
    K = len(lib["cluster"])
    log.info("Library: %d clusters, grid %d×%d", K, len(lat_gcm), len(lon_gcm))
    log.info("Calendar window: ±%d days", args.doy_window)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for ssp in args.ssps:
        out_path = Path(args.out_dir) / f"{args.gcm}_{ssp}_hourly.nc"
        if out_path.exists():
            log.info("%s: already exists — skipping", out_path.name)
            continue

        log.info("=== %s %s ===", args.gcm, ssp)

        # Load BC daily files
        def load_bc(vname):
            p = Path(args.bc_dir) / f"{vname}_{args.gcm}_{ssp}_bc.nc"
            da = xr.open_dataset(p)[vname]
            da = da.assign_coords(lat=lat_gcm, lon=lon_gcm)
            return da.load()

        rsds_d  = load_bc("rsds")    # (time, lat, lon)  W m-2 daily mean
        tas_d   = load_bc("tas")     # (time, lat, lon)  K     daily mean
        wind_d  = load_bc("sfcWind") # (time, lat, lon)  m s-1 daily mean

        times_d = pd.DatetimeIndex(rsds_d.time.values)
        n_days  = len(times_d)
        n_lat, n_lon = len(lat_gcm), len(lon_gcm)

        log.info("BC days loaded: %d", n_days)

        # Build feature matrix for GCM daily data
        X_gcm = np.stack([
            rsds_d.values,
            tas_d.values,
            wind_d.values,
        ], axis=-1)  # (n_days, lat, lon, 3)

        X_norm = (X_gcm - feat_mean[np.newaxis]) / feat_std[np.newaxis]

        # Cluster assignment with calendar-window filter (per day)
        log.info("Assigning clusters with DOY window ±%d …", args.doy_window)
        labels = np.empty((n_days, n_lat, n_lon), dtype=np.int32)
        for d in range(n_days):
            doy   = times_d[d].dayofyear
            valid = get_valid_clusters(doy, cluster_doys, args.doy_window)
            x_day = X_norm[d].reshape(-1, 3)                              # (lat*lon, 3)
            diff  = x_day[:, np.newaxis, :] - centroids[valid][np.newaxis]
            dist2 = (diff ** 2).sum(axis=-1)                              # (lat*lon, |valid|)
            labels[d] = valid[dist2.argmin(axis=-1)].reshape(n_lat, n_lon)
            if d % 365 == 0:
                log.info("  Day %d / %d  (DOY=%d, %d valid clusters)",
                         d, n_days, doy, len(valid))

        # Reconstruct hourly values
        log.info("Reconstructing hourly arrays …")
        rsds_hourly  = np.empty((n_days, 24, n_lat, n_lon), dtype=np.float32)
        tas_hourly   = np.empty((n_days, 24, n_lat, n_lon), dtype=np.float32)
        wind_hourly  = np.empty((n_days, 24, n_lat, n_lon), dtype=np.float32)

        rsds_v = rsds_d.values
        tas_v  = tas_d.values
        wind_v = wind_d.values

        for k in range(K):
            mask = (labels == k)  # (n_days, lat, lon)
            if not mask.any():
                continue
            # broadcast profiles over days dimension
            mask24 = mask[:, np.newaxis, :, :]  # (n_days, 1, lat, lon)
            pf = prof_frac[k]    # (24, lat, lon)
            pa = prof_anom[k]
            pr = prof_ratio[k]

            rsds_hourly  = np.where(
                mask24,
                rsds_v[:, np.newaxis, :, :] * pf[np.newaxis],
                rsds_hourly,
            )
            tas_hourly   = np.where(
                mask24,
                tas_v[:, np.newaxis, :, :] + pa[np.newaxis],
                tas_hourly,
            )
            wind_hourly  = np.where(
                mask24,
                wind_v[:, np.newaxis, :, :] * pr[np.newaxis],
                wind_hourly,
            )

        # Clip physical bounds
        rsds_hourly = np.maximum(rsds_hourly, 0.0)
        wind_hourly = np.maximum(wind_hourly, 0.0)

        # Build time index: one timestamp per hour
        time_hourly = pd.date_range(
            start=times_d[0],
            periods=n_days * 24,
            freq="h",
        )
        rsds_hourly_2d  = rsds_hourly.reshape(n_days * 24, n_lat, n_lon)
        tas_hourly_2d   = tas_hourly.reshape(n_days * 24, n_lat, n_lon)
        wind_hourly_2d  = wind_hourly.reshape(n_days * 24, n_lat, n_lon)

        ds_out = xr.Dataset({
            "rsds":    xr.DataArray(
                rsds_hourly_2d,
                dims=["time", "lat", "lon"],
                coords={"time": time_hourly, "lat": lat_gcm, "lon": lon_gcm},
                attrs={"units": "W m-2",
                       "long_name": "Surface downwelling shortwave radiation"},
            ),
            "tas":     xr.DataArray(
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
            "description": (f"Hourly downscaled BC GCM data — {args.gcm} {args.run} {ssp}"),
            "method":      "K-means diurnal profile downscaling from ERA5 hourly "
                           f"with calendar window ±{args.doy_window} days",
            "library":     str(args.library),
            "gcm":         args.gcm,
            "run":         args.run,
            "ssp":         ssp,
            "doy_window":  args.doy_window,
        }

        n_lat_out, n_lon_out = len(lat_gcm), len(lon_gcm)
        encoding = {
            v: {"zlib": True, "complevel": 4,
                "chunksizes": (24, n_lat_out, n_lon_out)}
            for v in ds_out.data_vars
        }
        ds_out.to_netcdf(out_path, encoding=encoding)
        log.info("→ %s", out_path.name)
        del rsds_d, tas_d, wind_d, rsds_hourly, tas_hourly, wind_hourly, ds_out

    log.info("All SSPs done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Hourly temporal downscaling of BC GCM daily data"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── fit ──────────────────────────────────────────────────────────────────
    fit = sub.add_parser("fit", help="Train K-means diurnal library from ERA5 hourly")
    fit.add_argument("--era5-nc",      required=True,
                     help="Glob pattern for ERA5 hourly NetCDF files produced by "
                          "convert_era5_hourly.py "
                          "(e.g. '/data/raw/era5/era5_india_*.nc')")
    fit.add_argument("--gcm-grid",     required=True, type=Path,
                     help="Any BC output NetCDF (used only for lat/lon grid)")
    fit.add_argument("--out-library",  required=True,
                     help="Output library NetCDF path")
    fit.add_argument("--n-clusters",   type=int, default=30,
                     help="Number of K-means clusters (default: 30)")
    fit.add_argument("--env-dir",      type=Path, default=None,
                     help="Conda env root for ESMFMKFILE (default: sys.prefix)")

    # ── apply ─────────────────────────────────────────────────────────────────
    app = sub.add_parser("apply", help="Apply diurnal library to BC GCM daily data")
    app.add_argument("--library",   required=True,
                     help="Library NetCDF produced by 'fit'")
    app.add_argument("--bc-dir",    required=True, type=Path,
                     help="Directory with bias-corrected daily NetCDF files")
    app.add_argument("--gcm",       default="CanESM5")
    app.add_argument("--run",       default="r10i1p1f1")
    app.add_argument("--ssps",      nargs="+", default=["ssp245", "ssp585"])
    app.add_argument("--out-dir",    required=True, type=Path,
                     help="Output directory for hourly NetCDF files")
    app.add_argument("--doy-window", type=int, default=30,
                     help="Calendar half-window in days for cluster selection (default: 30). "
                          "Only clusters whose circular-mean DOY is within this range of the "
                          "target day are considered.")

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "fit":
        cmd_fit(args)
    else:
        cmd_apply(args)


if __name__ == "__main__":
    main()
