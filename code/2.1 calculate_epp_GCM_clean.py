# -*- coding: cp1252 -*-
import os
os.environ['ESMFMKFILE'] = "/gpfs/workdir/shared/juicce/envs/xenv/lib/esmf.mk"
import xesmf as xe
import xarray as xr
import numpy as np
import pandas as pd
import glob as glob
import dask.array as da
import gc
from dataclasses import dataclass, field
from xclim import sdba
from dask import delayed, compute
import geopandas as gpd
import xagg as xa
from rasterio.features import geometry_mask
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patheffects import withStroke
from matplotlib.gridspec import GridSpec
import cmocean as cmo
from collections import Counter

# -------------------------
# EPP physical constants
# -------------------------

@dataclass
class EPPConfig:
    """
    Physical constants for the wind and solar potential calculations.
    
    Wind power curve parameters
    ---------------------------
    vr  : rated wind speed (m/s) — turbine reaches full output above this
    vci : cut-in wind speed (m/s) — turbine starts generating below this
    vco : cut-out wind speed (m/s) — turbine shuts down above this
    wind_height_exponent : Hellmann exponent for log-law extrapolation (10 m ? 80 m)
    
    Solar (PV) cell temperature model parameters (Huld et al.)
    ----------------------------------------------------------
    gamma : temperature coefficient of PV efficiency (K?¹)
    T_ref : reference cell temperature (°C)
    G_stc : irradiance at standard test conditions (W m?²)
    c_1..c_4 : coefficients for the NOCT-based cell temperature model
    """
    # Wind turbine curve
    vr:  float = 13.0
    vci: float = 3.5
    vco: float = 25.0
    wind_height_exponent: float = 0.143   # (80/10)^0.143

    # PV cell temperature
    gamma: float = -0.005
    T_ref: float = 25.0
    G_stc: float = 1000.0
    c_1: float =  4.3
    c_2: float =  0.943
    c_3: float =  0.028
    c_4: float = -1.528


# Default config instance — can be overridden at call sites
DEFAULT_EPP_CONFIG = EPPConfig()


# -------------------------
# Helper functions
# -------------------------

def load_variable(var, GCM, ssp, run, path_folder, gwl, chunks):
    """
    Load a single variable dataset from a file matching a pattern.
    Only keep the variable and lat, lon, time coordinates.
    Drop 'height' variable if present.
    """
    pattern = f"{path_folder}{GCM}/{var}_day_{GCM}_{ssp}_{run}*{gwl}.nc"
    files = glob.glob(pattern)
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    try:
        ds = xr.open_dataset(files[0], chunks=chunks)
    except Exception:
        ds = xr.open_dataset(files[0], chunks=chunks, decode_times=False)
        ds['time'] = pd.to_datetime(ds['time'], unit='D', origin='2015-01-01')
    ds['time'] = ds['time'].dt.floor('D')
    ds = ds[[var, 'lat', 'lon', 'time']]
    if 'height' in ds:
        ds = ds.drop_vars('height')
    return ds


def grids_match(ds1, ds2):
    """Check whether the lat/lon grids of two datasets match."""
    lat_match = (ds1.lat.shape[0] == ds2.lat.shape[0]
                 and np.array_equal(ds1.lat.values, ds2.lat.values))
    lon_match = (ds1.lon.shape[0] == ds2.lon.shape[0]
                 and np.array_equal(ds1.lon.values, ds2.lon.values))
    return lat_match and lon_match


def choose_target_grid(datasets, method="min"):
    """
    Choose a target grid dataset from a dictionary of datasets.
    'min'  ? dataset with the smallest lat/lon dimensions.
    'mode' ? dataset whose dimensions are the most common.
    """
    dims = [(ds.lat.shape[0], ds.lon.shape[0]) for ds in datasets.values()]
    if method == "min":
        target_lat = min(d[0] for d in dims)
        target_lon = min(d[1] for d in dims)
    elif method == "mode":
        lat_list = [d[0] for d in dims]
        lon_list = [d[1] for d in dims]
        target_lat = max(set(lat_list), key=lat_list.count)
        target_lon = max(set(lon_list), key=lon_list.count)
    else:
        raise ValueError(f"Unknown method for target grid selection: {method!r}")

    for ds in datasets.values():
        if ds.lat.shape[0] == target_lat and ds.lon.shape[0] == target_lon:
            return ds
    return list(datasets.values())[0]


def regrid_to_target(ds, target_ds, var_name):
    """Regrid ds to the target_ds grid if needed."""
    if not grids_match(ds, target_ds):
        print(f"Regridding {var_name}")
        regridder = xe.Regridder(ds, target_ds, method='bilinear')
        return regridder(ds)
    return ds


def rasterize_shapefile(shapefile, coords, shape, transform):
    """Rasterize shapefile geometries onto a grid defined by shape/transform."""
    geometries = shapefile['geometry']
    mask = geometry_mask(
        geometries=geometries,
        all_touched=True,
        out_shape=shape,
        transform=transform,
        invert=True
    )
    return mask


def safe_to_netcdf(ds, path, mode="w", **kwargs):
    """Write ds to a temporary file, then atomically rename to path."""
    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(tmp_path, mode=mode, **kwargs)
    os.replace(tmp_path, path)


# -------------------------
# Main loading function
# -------------------------

def load_ds(GCM, ssp, run, path_folder, gwl):
    """
    Load tasmax, tas, rsds, and either (uas, vas) or sfcWind.
    Regrid to a common grid, merge, and convert the calendar.
    """
    chunks = {'time': -1, 'lat': 100, 'lon': 100}

    dtasmax = load_variable('tasmax', GCM, ssp, run, path_folder, gwl, chunks)
    dtas    = load_variable('tas',    GCM, ssp, run, path_folder, gwl, chunks)
    drsds   = load_variable('rsds',   GCM, ssp, run, path_folder, gwl, chunks)

    datasets = {'tasmax': dtasmax, 'tas': dtas, 'rsds': drsds}

    uas_pattern = f"{path_folder}{GCM}/uas_*{GCM}_{ssp}_{run}*{gwl}.nc"
    if glob.glob(uas_pattern):
        duas = load_variable('uas', GCM, ssp, run, path_folder, gwl, chunks)
        dvas = load_variable('vas', GCM, ssp, run, path_folder, gwl, chunks)
        datasets.update({'uas': duas, 'vas': dvas})
        target_ds = choose_target_grid(datasets, method="min")
        for key, ds in datasets.items():
            datasets[key] = regrid_to_target(ds, target_ds, key)
        ds = xr.merge([datasets['tasmax'], datasets['tas'],
                       datasets['rsds'], datasets['uas'], datasets['vas']])
        ds['sfcWind'] = np.sqrt(ds.uas**2 + ds.vas**2)
        ds = ds.drop_vars(['uas', 'vas'])
    else:
        dsfcWind = load_variable('sfcWind', GCM, ssp, run, path_folder, gwl, chunks)
        datasets['sfcWind'] = dsfcWind
        target_ds = choose_target_grid(datasets, method="mode")
        for key, ds_item in datasets.items():
            datasets[key] = regrid_to_target(ds_item, target_ds, key)
        ds = xr.merge(list(datasets.values()))

    try:
        ds = ds.convert_calendar('standard')
    except Exception:
        ds = ds.convert_calendar('standard', align_on='year')
        ds = ds.interp(time=pd.date_range(ds.time.values[0], ds.time.values[-1], freq='D'))

    ds = ds.sortby('lat').sortby('lon').sortby('time')
    ds = ds.chunk(chunks)
    return ds


def set_variable_units(ds, var_units):
    """Set the 'units' attribute for each variable listed in var_units."""
    for var, units in var_units.items():
        if var in ds:
            ds[var].attrs['units'] = units
    return ds


def filter_domain(ds, lat_range, lon_range):
    """Subset ds to the given lat/lon ranges."""
    return ds.sel(lat=slice(lat_range[0], lat_range[1]),
                  lon=slice(lon_range[0], lon_range[1]))


def get_output_filename(path_preprocessed, GCM, ssp, run, gwl, reanalysis):
    """Return the expected bias-corrected output filename."""
    folder = os.path.join(path_preprocessed, GCM)
    fname = f"dadjusted_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.nc"
    return os.path.join(folder, fname)


# -------------------------
# Bias correction
# -------------------------

def unbias_GCM(GCM, run, ssp, path_preprocessed, shapefile_path, path_folder, gwl_list,
               reanalysis='W5E5', overwrite=False):
    """
    Train an MBCn adjustment on historical data and apply it to each future GWL.

    Each GWL is processed as a true Dask delayed task; all tasks are submitted
    before any computation starts, so they can run in parallel on the cluster.
    """
    print("Starting unbias_GCM function")

    gwl_unbias = []
    for gwl in gwl_list:
        outfile = get_output_filename(path_preprocessed, GCM, ssp, run, gwl, reanalysis)
        if os.path.exists(outfile) and not overwrite:
            print(f"{gwl}: File already exists")
        else:
            gwl_unbias.append(gwl)

    if not gwl_unbias:
        print("All files already exist. Exiting function.")
        return

    # ------------------------------------------------------------------
    # 1. Historical simulation (GWL0-61)
    # ------------------------------------------------------------------
    dhist = load_ds(GCM, ssp, run, path_folder, 'GWL0-61').dropna('time', how='all')
    chunk_loc = 60

    files_ref = glob.glob(os.path.join(path_folder, reanalysis, f"*{reanalysis}*.nc"))
    dref = xr.open_mfdataset(files_ref)
    dref = dref.sortby('lat').sortby('lon').sortby('time')
    dhist = dhist.sortby('lat').sortby('lon').sortby('time')
    dref = dref.chunk({'time': -1, 'lat': 50, 'lon': 50})

    lat_range = (dref.lat.values[0], dref.lat.values[-1])
    lon_range = (dref.lon.values[0], dref.lon.values[-1])

    dhist = filter_domain(dhist, lat_range, lon_range)
    dhist = dhist.chunk({'time': -1, 'lat': 20, 'lon': 20})

    # --- Build the raster mask from the shapefile ---
    mask_template = dref.tas.isel(time=0).load()
    shapefile = gpd.read_file(shapefile_path)
    lons, lats = np.meshgrid(mask_template.lon, mask_template.lat)
    coords = np.array([lons.flatten(), lats.flatten()]).T
    transform = rasterio.transform.from_bounds(
        mask_template.lon.min().item(), mask_template.lat.min().item(),
        mask_template.lon.max().item(), mask_template.lat.max().item(),
        len(mask_template.lon), len(mask_template.lat)
    )
    mask = rasterize_shapefile(shapefile, coords, mask_template.shape, transform)
    mask = mask[::-1, :]

    dref = dref.where(mask == 1, np.nan)
    dref['mask'] = xr.where(~np.isnan(dref.isel(time=0).tas), 1, 0)
    print("dref mask coverage:", float(dref['mask'].sum() / dref['mask'].count()))

    # Regrid reference to dhist grid
    regridder = xe.Regridder(dref, dhist, method='conservative_normed')
    dref = regridder(dref, output_chunks={'lat': 50, 'lon': 50})
    dref = dref.convert_calendar('noleap').convert_calendar('standard')
    dhist = dhist.convert_calendar('noleap').convert_calendar('standard')

    # Second, finer mask on the regridded grid
    ref_grid = dref.tas.isel(time=0)

    def create_mask_from_shapefile(grid, shapefile):
        transform = rasterio.transform.from_bounds(
            grid.lon.min().item(), grid.lat.min().item(),
            grid.lon.max().item(), grid.lat.max().item(),
            len(grid.lon), len(grid.lat)
        )
        shape = (len(grid.lat), len(grid.lon))
        mask = geometry_mask(
            geometries=shapefile.geometry,
            all_touched=True,
            out_shape=shape,
            transform=transform,
            invert=True
        )
        return xr.DataArray(
            mask[::-1, :], dims=("lat", "lon"),
            coords={"lat": grid.lat, "lon": grid.lon}
        )

    mask_array = create_mask_from_shapefile(ref_grid, shapefile)

    var_units = {'sfcWind': 'm s-1', 'tas': 'K', 'tasmax': 'K', 'rsds': 'W m-2'}
    dref = dref.where(mask_array)
    dhist = dhist.where(mask_array)
    dref = set_variable_units(dref, var_units)
    dhist = set_variable_units(dhist, var_units)

    dref = dref.sel(time=slice('1982-01-01', '2001-12-31'))
    dref = dref[['sfcWind', 'tas', 'tasmax', 'rsds']]

    lon_ori = dref.lon
    lat_ori = dref.lat

    dref = dref.stack(location=("lat", "lon"))
    dhist = dhist.stack(location=("lat", "lon"))

    # Jitter lower bounds — set to a fixed safe value
    rsds_low = 1e-6
    wind_low = 1e-6

    def remove_constant_locations(da, dim='time'):
        for v in da.data_vars:
            if dim in da[v].dims:
                std = da[v].std(dim=dim)
                is_const = (std == 0).compute()
                if is_const.any():
                    idx_to_remove = da.location[is_const]
                    print(f"Variable '{v}': removing {len(idx_to_remove)} constant locations.")
                    da = da.drop_sel(location=idx_to_remove)
        return da

    valid_mask = (~dref.tas.isnull().all('time')).compute()
    nb = valid_mask.count()
    print("Valid fraction before cleaning:", float(valid_mask.sum() / nb))
    dref = remove_constant_locations(dref)
    dref = dref.dropna(dim='location', how='all')
    print("Remaining locations:", dref.location.shape[0])

    dhist = dhist.sel(location=dref.location).sortby('location')
    dref = dref.sortby('location')

    for ds_name, ds_obj in [('dref', dref), ('dhist', dhist)]:
        ds_obj = ds_obj.assign(
            rsds=sdba.processing.to_additive_space(
                sdba.processing.jitter(ds_obj.rsds,
                                       lower=f"{rsds_low} W m-2", minimum="0 W m-2"),
                lower_bound="0 W m-2", trans="log",
            ),
            sfcWind=sdba.processing.to_additive_space(
                sdba.processing.jitter(ds_obj.sfcWind,
                                       lower=f"{wind_low} m s-1", minimum="0 m s-1"),
                lower_bound="0 m s-1", trans="log",
            )
        )
        if ds_name == 'dref':
            dref = ds_obj
        else:
            dhist = ds_obj

    loc_values = dref.location.to_dataframe().reset_index(drop=True)
    loc_values['location_index'] = loc_values.index

    ref  = dref.drop_vars(['lat', 'lon'])
    hist = dhist.drop_vars(['lat', 'lon'])
    ref  = sdba.processing.stack_variables(ref)
    hist = sdba.processing.stack_variables(hist)

    # Align historical time axis onto the reference period
    hist = hist.assign_coords(time=hist.time - hist.time.values[-1] + ref.time.values[-1])
    common_times = np.intersect1d(ref.time.values, hist.time.values)
    hist = hist.where(hist.time.isin(common_times))
    ref  = ref.where(ref.time.isin(common_times))

    ref  = ref.chunk({'time': -1, 'location': chunk_loc})
    hist = hist.chunk({'time': -1, 'location': chunk_loc})

    # Train once
    ADJ = sdba.MBCn.train(
        ref, hist,
        base_kws={"nquantiles": 30, "group": "time"},
        adj_kws={"interp": "nearest", "extrapolation": "constant"},
        n_iter=20,
        n_escore=1000,
        pts_dim='multivar',
    )

    # Load all future datasets eagerly so they are available inside delayed tasks
    dfut_datasets = {
        gwl: load_ds(GCM, ssp, run, path_folder, gwl) for gwl in gwl_unbias
    }

    # ------------------------------------------------------------------
    # 2. Process each future GWL as a *true* Dask delayed task
    # ------------------------------------------------------------------

    @delayed
    def process_gwl_delayed(dfut, dref, gwl, ADJ, ref, hist, loc_values, mask_array,
                             GCM, ssp, run, path_preprocessed, reanalysis,
                             lat_ori, lon_ori, rsds_low, wind_low, chunk_loc):
        """
        Bias-correct a single future GWL dataset and write it to disk.
        Decorated with @delayed so all GWLs can be submitted simultaneously
        and executed in parallel by the Dask scheduler.

        Note: the time axis of `fut` is shifted onto the reference period
        because MBCn requires training and simulation data to share the same
        calendar positions (the adjustment is purely distributional, not
        chronological).
        """
        print(f"[Delayed] Processing GWL: {gwl}")

        dfut = filter_domain(dfut, (lat_ori[0], lat_ori[-1]), (lon_ori[0], lon_ori[-1]))
        dfut = set_variable_units(dfut,
                                  {'sfcWind': 'm s-1', 'tas': 'K', 'tasmax': 'K', 'rsds': 'W m-2'})
        dfut = dfut.convert_calendar('noleap').convert_calendar('standard')
        dfut = dfut.sortby('lat').sortby('lon').sortby('time')
        dfut = dfut.stack(location=("lat", "lon"))

        dfut = dfut.assign(
            rsds=sdba.processing.to_additive_space(
                sdba.processing.jitter(dfut.rsds,
                                       lower=f"{rsds_low} W m-2", minimum="0 W m-2"),
                lower_bound="0 W m-2", trans="log",
            ),
            sfcWind=sdba.processing.to_additive_space(
                sdba.processing.jitter(dfut.sfcWind,
                                       lower=f"{wind_low} m s-1", minimum="0 m s-1"),
                lower_bound="0 m s-1", trans="log",
            )
        )

        # Intersect times between ref/hist and fut
        common_times = np.intersect1d(ref.time.values, hist.time.values)
        dref_t = dref.sel(time=common_times)
        dfut   = dfut.sel(time=common_times)

        # Align on the location dimension
        dref_t, dfut = xr.align(dref_t, dfut, join="inner", copy=False)

        dfut = dfut.drop_vars(['lat', 'lon'], errors='ignore')
        fut  = sdba.processing.stack_variables(dfut)

        # Shift fut time axis onto the reference period (required by MBCn)
        fut = fut.assign_coords(time=fut.time - fut.time.values[-1] + ref.time.values[-1])
        fut = fut.chunk({'time': -1, 'location': chunk_loc})

        adj = ADJ.adjust(
            ref=ref,
            hist=hist,
            sim=fut,
            base=sdba.QuantileDeltaMapping,
            adj_kws={"interp": "linear", "extrapolation": "constant"},
        )

        adj = sdba.unstack_variables(adj).compute()
        adj = adj.assign(
            rsds=sdba.processing.from_additive_space(adj.rsds),
            sfcWind=sdba.processing.from_additive_space(adj.sfcWind)
        )

        adj = adj.assign_coords(location=loc_values['location'])
        adj = (adj
               .assign_coords(lat=("location", loc_values["lat"].values),
                               lon=("location", loc_values["lon"].values))
               .set_index(location=["lat", "lon"])
               .unstack("location"))

        adj = adj.reindex(lat=lat_ori, lon=lon_ori)
        adj.rsds.values[adj.rsds.values < 1e-5] = 0
        adj.sfcWind.values[adj.sfcWind.values < 1e-5] = 0

        for var in adj.data_vars:
            adj[var] = adj[var].astype(np.float32)

        out_file = get_output_filename(path_preprocessed, GCM, ssp, run, gwl, reanalysis)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        adj.to_netcdf(out_file)
        print(f"[Delayed] Saved: {out_file}")

    # Build the list of delayed tasks — nothing runs yet
    tasks = [
        process_gwl_delayed(
            dfut_datasets[gwl], dref, gwl, ADJ, ref, hist,
            loc_values, mask_array, GCM, ssp, run,
            path_preprocessed, reanalysis,
            lat_ori, lon_ori, rsds_low, wind_low, chunk_loc
        )
        for gwl in gwl_unbias
    ]

    # Trigger parallel execution of all tasks
    print(f"Submitting {len(tasks)} delayed task(s) to the Dask scheduler...")
    compute(*tasks)
    print("All GWL tasks completed.")


# -------------------------
# EPP calculation
# -------------------------

def calculate_epp_reanalysis_grid_GCM(
    GCM, run, ssp,
    path_preprocessed, path_folder,
    reanalysis='W5E5',
    shapefile_path=None,
    cfg: EPPConfig = DEFAULT_EPP_CONFIG,
):
    """Compute wpp/spp reference files from reanalysis regridded to the GCM grid."""

    out_folder = os.path.join(path_preprocessed, GCM)
    os.makedirs(out_folder, exist_ok=True)
    path_wpp_ref = os.path.join(out_folder, f"wpp_ref_{GCM}_{reanalysis}.nc")
    path_spp_ref = os.path.join(out_folder, f"spp_ref_{GCM}_{reanalysis}.nc")

    if os.path.exists(path_wpp_ref) and os.path.exists(path_spp_ref):
        print("EPP ref files already exist, skipping.")
        return

    # 1. Load GCM grid template
    gcm_file = os.path.join(
        path_preprocessed, GCM,
        f"wpp_day_{GCM}_{ssp}_{run}_GWL0-61_{reanalysis}.nc"
    )
    print("Loading GCM file:", gcm_file)
    ds_gcm = xr.open_dataset(gcm_file, chunks={'time': 100})
    gcm_grid = xr.Dataset(coords={"lat": ds_gcm["lat"], "lon": ds_gcm["lon"]})

    # 2. Load reanalysis
    files_ref = glob.glob(os.path.join(path_folder, reanalysis, f"*{reanalysis}*.nc"))
    if not files_ref:
        raise FileNotFoundError(
            f"No reanalysis files found in {os.path.join(path_folder, reanalysis)}")

    print(f"Found {len(files_ref)} reanalysis files for {reanalysis}")
    dref = xr.open_mfdataset(
        files_ref, combine='by_coords',
        chunks={'time': -1, 'lat': 100, 'lon': 100}, parallel=True
    )
    dref = dref.sortby('lat').sortby('lon').sortby('time')

    for v in ('tas', 'tasmax', 'rsds'):
        if v not in dref:
            raise KeyError(f"{v} not found in reanalysis dataset")

    if 'sfcWind' not in dref:
        if not {'uas', 'vas'}.issubset(dref.data_vars):
            raise KeyError("Need either 'sfcWind' or both 'uas' and 'vas' in reanalysis")
        print("Computing sfcWind from uas/vas")
        dref['sfcWind'] = np.hypot(dref['uas'], dref['vas'])

    dref = dref[['tas', 'tasmax', 'rsds', 'sfcWind']]
    dref = dref.chunk({'time': -1, 'lat': 50, 'lon': 50})

    # 3. Optional shapefile mask (before regrid)
    if shapefile_path is not None:
        print("Applying shapefile mask:", shapefile_path)
        shapefile = gpd.read_file(shapefile_path)
        mask_template = dref.tas.isel(time=0).load()
        lons, lats = np.meshgrid(mask_template.lon, mask_template.lat)
        coords = np.array([lons.flatten(), lats.flatten()]).T
        transform = rasterio.transform.from_bounds(
            float(mask_template.lon.min()), float(mask_template.lat.min()),
            float(mask_template.lon.max()), float(mask_template.lat.max()),
            len(mask_template.lon), len(mask_template.lat)
        )
        mask = rasterize_shapefile(shapefile, coords, mask_template.shape, transform)
        mask = mask[::-1, :]
        dref = dref.where(mask == 1, np.nan)
        mask_da = xr.where(~np.isnan(mask_template), mask, 0)
        dref['mask'] = mask_da
        print("Fraction of grid kept after mask:",
              float(dref['mask'].sum() / dref['mask'].count()))
        dref = dref.drop_vars('mask')

    # 4. Regrid to GCM grid
    print("Regridding reanalysis to GCM grid...")
    regridder = xe.Regridder(dref, gcm_grid, method='conservative_normed', reuse_weights=False)
    dref_rg = regridder(dref, output_chunks={'lat': 50, 'lon': 50})
    ds_gcm.close()

    dref_rg = dref_rg.convert_calendar('noleap').convert_calendar('standard')
    dref_rg['tas']    = dref_rg['tas']    - 273.15
    dref_rg['tasmax'] = dref_rg['tasmax'] - 273.15

    std_mask = dref_rg.tas.std(dim='time')
    if hasattr(std_mask, 'compute'):
        std_mask = std_mask.compute()
    dref_rg = dref_rg.where(~std_mask.isnull() & (std_mask != 0), drop=True)

    # 5. Solar potential (spp)
    print("Computing solar potential (spp)...")
    T_cell = (cfg.c_1
              + cfg.c_2 * ((dref_rg['tasmax'] + dref_rg['tas']) / 2)
              + cfg.c_3 * dref_rg['rsds']
              + cfg.c_4 * dref_rg['sfcWind'])
    P_R = 1 + cfg.gamma * (T_cell - cfg.T_ref)
    spp = P_R * (dref_rg['rsds'] / cfg.G_stc)

    solar_potential = spp.to_dataset(name='spp').convert_calendar('noleap')
    solar_potential = solar_potential.chunk({'time': 100, 'lat': -1, 'lon': -1})
    solar_potential['spp'] = solar_potential['spp'].astype('f4')
    solar_potential.attrs.update({
        'DESCRIPTION': f'spp reference for {reanalysis} regridded to {GCM} grid',
        'units': 'dimensionless',
        'long_name': 'PVtot potential',
        'SOURCE': 'calculate_epp_reanalysis_grid_GCM',
        'AUTHOR': 'Colin Lenoble',
    })
    solar_potential = solar_potential.compute()
    safe_to_netcdf(solar_potential, path_spp_ref)
    print("Written spp to", path_spp_ref)

    # 6. Wind potential (wpp)
    print("Computing wind potential (wpp)...")
    wind_pot = dref_rg['sfcWind'] * (80.0 / 10.0) ** cfg.wind_height_exponent
    wind_pot = xr.where(wind_pot < cfg.vci, 0, wind_pot)
    wind_pot = xr.where(wind_pot >= cfg.vco, 0, wind_pot)
    wind_pot = xr.where((wind_pot >= cfg.vr) & (wind_pot < cfg.vco), 1, wind_pot)
    wind_pot = xr.where(
        (wind_pot >= cfg.vci) & (wind_pot < cfg.vr),
        (wind_pot**3 - cfg.vci**3) / (cfg.vr**3 - cfg.vci**3),
        wind_pot
    )

    wind_potential = wind_pot.to_dataset(name='wpp')
    wind_potential = wind_potential.chunk({'time': 100, 'lat': -1, 'lon': -1})
    wind_potential['wpp'] = wind_potential['wpp'].astype('f4')
    wind_potential.attrs.update({
        'DESCRIPTION': f'wpp reference for {reanalysis} regridded to {GCM} grid',
        'units': 'dimensionless',
        'long_name': 'Wind potential',
        'SOURCE': 'calculate_epp_reanalysis_grid_GCM',
        'AUTHOR': 'Colin Lenoble',
    })
    wind_potential = wind_potential.compute()
    safe_to_netcdf(wind_potential, path_wpp_ref)
    print("Written wpp to", path_wpp_ref)

    solar_potential.close()
    wind_potential.close()
    dref_rg.close()
    dref.close()
    gc.collect()
    print("EPP reference files saved:", path_spp_ref, path_wpp_ref)


def calculate_epp_GCM(GCM, run, ssp, path_preprocessed, gwl,
                      reanalysis='W5E5', cfg: EPPConfig = DEFAULT_EPP_CONFIG):
    """Compute wpp/spp from a bias-corrected GCM file."""

    ds_path = os.path.join(path_preprocessed, GCM,
                           f"dadjusted_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.nc")
    ds = xr.open_dataset(ds_path)

    wpp_path = os.path.join(path_preprocessed, GCM,
                            f"wpp_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.nc")
    spp_path = os.path.join(path_preprocessed, GCM,
                            f"spp_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.nc")

    if os.path.exists(wpp_path) and os.path.exists(spp_path):
        print('EPP files already exist')
        return

    print('Calculating EPP')
    ds = ds.convert_calendar('noleap')
    ds['tasmax'] = ds['tasmax'] - 273.15
    ds['tas']    = ds['tas']    - 273.15

    # Solar potential
    T_cell = (cfg.c_1
              + cfg.c_2 * ((ds['tasmax'] + ds['tas']) / 2)
              + cfg.c_3 * ds['rsds']
              + cfg.c_4 * ds['sfcWind'])
    P_R = 1 + cfg.gamma * (T_cell - cfg.T_ref)
    solar_potential = P_R * (ds['rsds'] / cfg.G_stc)

    solar_xr = solar_potential.to_dataset(name='spp').convert_calendar('noleap')
    solar_xr = solar_xr.chunk({'time': 100, 'lat': -1, 'lon': -1})
    solar_xr['spp'] = solar_xr['spp'].astype('f4')
    solar_xr.attrs.update({
        'DESCRIPTION': f"{GCM} solar potential",
        'units': 'dimensionless', 'long_name': 'PVtot potential',
        'SOURCE': 'calculate_epp_GCM_dask.py', 'AUTHOR': 'Colin Lenoble', 'corrected': 1,
    })
    solar_xr.to_netcdf(spp_path, mode='w')

    # Wind potential
    wind_pot = ds['sfcWind'] * (80 / 10) ** cfg.wind_height_exponent
    wind_pot = xr.where(wind_pot < cfg.vci, 0, wind_pot)
    wind_pot = xr.where(wind_pot >= cfg.vco, 0, wind_pot)
    wind_pot = xr.where((wind_pot >= cfg.vr) & (wind_pot < cfg.vco), 1, wind_pot)
    wind_pot = xr.where((wind_pot >= cfg.vci) & (wind_pot < cfg.vr),
                        (wind_pot**3 - cfg.vci**3) / (cfg.vr**3 - cfg.vci**3),
                        wind_pot)

    wind_xr = wind_pot.to_dataset(name='wpp')
    wind_xr = wind_xr.chunk({'time': 100, 'lat': -1, 'lon': -1})
    wind_xr['wpp'] = wind_xr['wpp'].astype('f4')
    wind_xr.attrs.update({
        'DESCRIPTION': f"{GCM} wind potential",
        'units': 'dimensionless', 'long_name': 'Wind potential',
        'SOURCE': 'calculate_epp_GCM_dask.py', 'AUTHOR': 'Colin Lenoble', 'corrected': 1,
    })
    wind_xr.to_netcdf(wpp_path, mode='w')

    print('EPP saved')
    solar_xr.close()
    wind_xr.close()
    del solar_xr, wind_xr
    gc.collect()


def calculate_epp_reanalysis(
    path_folder,
    path_preprocessed,
    reanalysis='W5E5',
    shapefile_path=None,
    cfg: EPPConfig = DEFAULT_EPP_CONFIG,
):
    """
    Compute wpp/spp on the **native reanalysis grid** (no GCM regridding).

    Loads the raw reanalysis files, optionally applies a shapefile mask,
    converts units, and writes the EPP datasets to disk.

    Outputs
    -------
    {path_preprocessed}/{reanalysis}/wpp_{reanalysis}.nc
    {path_preprocessed}/{reanalysis}/spp_{reanalysis}.nc
    """
    out_folder = os.path.join(path_preprocessed, reanalysis)
    os.makedirs(out_folder, exist_ok=True)
    path_wpp = os.path.join(out_folder, f"wpp_day_{reanalysis}_historical_reanalysis_19790101-20191231.nc")
    path_spp = os.path.join(out_folder, f"spp_day_{reanalysis}_historical_reanalysis_19790101-20191231.nc")

    if os.path.exists(path_wpp) and os.path.exists(path_spp):
        print("Reanalysis EPP files already exist, skipping.")
        return

    # 1. Load reanalysis files
    files_ref = glob.glob(os.path.join(path_folder, reanalysis, f"*{reanalysis}*.nc"))
    if not files_ref:
        raise FileNotFoundError(
            f"No reanalysis files found in {os.path.join(path_folder, reanalysis)}")
    print(f"Found {len(files_ref)} reanalysis files for {reanalysis}")

    dref = xr.open_mfdataset(
        files_ref, combine='by_coords',
        chunks={'time': -1, 'lat': 100, 'lon': 100}, parallel=True,
    )
    dref = dref.sortby('lat').sortby('lon').sortby('time')

    for v in ('tas', 'tasmax', 'rsds'):
        if v not in dref:
            raise KeyError(f"'{v}' not found in reanalysis dataset")

    if 'sfcWind' not in dref:
        if not {'uas', 'vas'}.issubset(dref.data_vars):
            raise KeyError("Need either 'sfcWind' or both 'uas' and 'vas' in reanalysis")
        print("Computing sfcWind from uas/vas")
        dref['sfcWind'] = np.hypot(dref['uas'], dref['vas'])

    dref = dref[['tas', 'tasmax', 'rsds', 'sfcWind']]
    dref = dref.chunk({'time': -1, 'lat': 50, 'lon': 50})

    # 2. Optional shapefile mask
    if shapefile_path is not None:
        print("Applying shapefile mask:", shapefile_path)
        shapefile = gpd.read_file(shapefile_path)
        mask_template = dref.tas.isel(time=0).load()
        transform = rasterio.transform.from_bounds(
            float(mask_template.lon.min()), float(mask_template.lat.min()),
            float(mask_template.lon.max()), float(mask_template.lat.max()),
            len(mask_template.lon), len(mask_template.lat),
        )
        lons, lats = np.meshgrid(mask_template.lon, mask_template.lat)
        coords = np.array([lons.flatten(), lats.flatten()]).T
        mask = rasterize_shapefile(shapefile, coords, mask_template.shape, transform)
        mask = mask[::-1, :]
        dref = dref.where(mask == 1, np.nan)
        print("Fraction of grid kept after mask:",
              float((mask == 1).sum() / mask.size))

    # 3. Unit conversion (K ? °C)
    dref = dref.convert_calendar('noleap').convert_calendar('standard')
    dref['tas']    = dref['tas']    - 273.15
    dref['tasmax'] = dref['tasmax'] - 273.15

    # 4. Solar potential (spp)
    print("Computing solar potential (spp)...")
    T_cell = (cfg.c_1
              + cfg.c_2 * ((dref['tasmax'] + dref['tas']) / 2)
              + cfg.c_3 * dref['rsds']
              + cfg.c_4 * dref['sfcWind'])
    P_R = 1 + cfg.gamma * (T_cell - cfg.T_ref)
    spp = P_R * (dref['rsds'] / cfg.G_stc)

    solar_potential = spp.to_dataset(name='spp').convert_calendar('noleap')
    solar_potential = solar_potential.chunk({'time': 100, 'lat': -1, 'lon': -1})
    solar_potential['spp'] = solar_potential['spp'].astype('f4')
    solar_potential.attrs.update({
        'DESCRIPTION': f'spp on native {reanalysis} grid',
        'units': 'dimensionless',
        'long_name': 'PVtot potential',
        'SOURCE': 'calculate_epp_reanalysis',
        'AUTHOR': 'Colin Lenoble',
    })
    solar_potential = solar_potential.compute()
    safe_to_netcdf(solar_potential, path_spp)
    print("Written spp to", path_spp)

    # 5. Wind potential (wpp)
    print("Computing wind potential (wpp)...")
    wind_pot = dref['sfcWind'] * (80.0 / 10.0) ** cfg.wind_height_exponent
    wind_pot = xr.where(wind_pot < cfg.vci, 0, wind_pot)
    wind_pot = xr.where(wind_pot >= cfg.vco, 0, wind_pot)
    wind_pot = xr.where((wind_pot >= cfg.vr) & (wind_pot < cfg.vco), 1, wind_pot)
    wind_pot = xr.where(
        (wind_pot >= cfg.vci) & (wind_pot < cfg.vr),
        (wind_pot**3 - cfg.vci**3) / (cfg.vr**3 - cfg.vci**3),
        wind_pot,
    )

    wind_potential = wind_pot.to_dataset(name='wpp')
    wind_potential = wind_potential.chunk({'time': 100, 'lat': -1, 'lon': -1})
    wind_potential['wpp'] = wind_potential['wpp'].astype('f4')
    wind_potential.attrs.update({
        'DESCRIPTION': f'wpp on native {reanalysis} grid',
        'units': 'dimensionless',
        'long_name': 'Wind potential',
        'SOURCE': 'calculate_epp_reanalysis',
        'AUTHOR': 'Colin Lenoble',
    })
    wind_potential = wind_potential.compute()
    safe_to_netcdf(wind_potential, path_wpp)
    print("Written wpp to", path_wpp)

    solar_potential.close()
    wind_potential.close()
    dref.close()
    gc.collect()
    print("Reanalysis EPP files saved:", path_spp, path_wpp)


# -------------------------
# Spatial aggregation
# -------------------------
def aggregate_epp(GCM, run, ssp, path_preprocessed, temp_folder, gwl, shapefile_path,
                  reanalysis='W5E5', suffix_shp='v1'):
    """
    Aggregate wind and solar potential by region.

    suffix_shp controls the weighting scheme:
      'v1' : weighted by climate reference capacity factors
             (mean spp/wpp over the reference period 1982-2001)
      'v2' : weighted by grid-cell area only (pure geographic weighting,
             xa.pixel_overlaps without explicit weights)

    Any other value raises a ValueError.
    """
    if suffix_shp not in ('v1', 'v2'):
        raise ValueError(
            f"Unknown suffix_shp {suffix_shp!r}. "
            "Expected 'v1' (capacity-factor weighted) or 'v2' (area weighted)."
        )

    wpp_path = f"{path_preprocessed}{GCM}/wpp_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.nc"
    spp_path = f"{path_preprocessed}{GCM}/spp_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.nc"
    wpp = xr.open_dataset(wpp_path)
    spp = xr.open_dataset(spp_path)
    shapefile = gpd.read_file(shapefile_path)

    wpp_ref = xr.open_dataset(
        f"{path_preprocessed}{GCM}/wpp_ref_{GCM}_{reanalysis}.nc"
    ).sel(time=slice('1982-01-01', '2001-12-31'))
    spp_ref = xr.open_dataset(
        f"{path_preprocessed}{GCM}/spp_ref_{GCM}_{reanalysis}.nc"
    ).sel(time=slice('1982-01-01', '2001-12-31'))

    wpp_ref = wpp_ref.sel(lat=slice(wpp.lat.values[0], wpp.lat.values[-1]),
                          lon=slice(wpp.lon.values[0], wpp.lon.values[-1]))
    spp_ref = spp_ref.sel(lat=slice(spp.lat.values[0], spp.lat.values[-1]),
                          lon=slice(spp.lon.values[0], spp.lon.values[-1]))

    os.makedirs(f"{temp_folder}{GCM}/", exist_ok=True)

    # ------------------------------------------------------------------
    # Build or load weight maps — one pair per weighting scheme
    # Weight map filenames include the suffix so v1 and v2 never collide
    # ------------------------------------------------------------------
    spp_wm_path = f"{temp_folder}{GCM}/{GCM}_weightmap_spp_{suffix_shp}"
    wpp_wm_path = f"{temp_folder}{GCM}/{GCM}_weightmap_wpp_{suffix_shp}"

    if suffix_shp == 'v1':
        # v1: weight each pixel by its mean capacity factor over 1982-2001
        if not os.path.exists(spp_wm_path):
            spp_weight_map = xa.pixel_overlaps(spp_ref, shapefile,
                                               weights=spp_ref.spp.mean(dim='time'))
            spp_weight_map.to_file(spp_wm_path)
        else:
            spp_weight_map = xa.read_wm(spp_wm_path)

        if not os.path.exists(wpp_wm_path):
            wpp_weight_map = xa.pixel_overlaps(wpp_ref, shapefile,
                                               weights=wpp_ref.wpp.mean(dim='time'))
            wpp_weight_map.to_file(wpp_wm_path)
        else:
            wpp_weight_map = xa.read_wm(wpp_wm_path)

    elif suffix_shp == 'v2':
        # v2: weight pixels by grid-cell area only (no capacity factor)
        if not os.path.exists(spp_wm_path):
            spp_weight_map = xa.pixel_overlaps(spp_ref, shapefile)
            spp_weight_map.to_file(spp_wm_path)
        else:
            spp_weight_map = xa.read_wm(spp_wm_path)

        if not os.path.exists(wpp_wm_path):
            wpp_weight_map = xa.pixel_overlaps(wpp_ref, shapefile)
            wpp_weight_map.to_file(wpp_wm_path)
        else:
            wpp_weight_map = xa.read_wm(wpp_wm_path)

    # ------------------------------------------------------------------
    # Attrs description — shared across solar and wind blocks
    # ------------------------------------------------------------------
    weighting_desc = (
        'weighted by mean solar/wind capacity factor over reference period 1982-2001'
        if suffix_shp == 'v1' else
        'weighted by grid-cell area only (no capacity factor)'
    )

    # ------------------------------------------------------------------
    # Solar aggregation
    # ------------------------------------------------------------------
    agg_solar = xa.aggregate(spp.load(), spp_weight_map).to_dataset()
    agg_solar.attrs.update({
        'units': 'dimensionless',
        'long_name': 'PVtot potential by country/region',
        'weighting': weighting_desc,
        'SOURCE': 'aggregate_epp',
        'AUTHOR': 'Colin Lenoble',
    })
    agg_solar.to_netcdf(
        f"{path_preprocessed}{GCM}/spp_agg_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}_{suffix_shp}.nc")

    # Reference aggregation — v1 only
    if suffix_shp == 'v1':
        spp_ref_agg_path = f"{path_preprocessed}{GCM}/spp_agg_ref_{GCM}_{reanalysis}.nc"
        if not os.path.exists(spp_ref_agg_path):
            agg_solar_ref = xa.aggregate(spp_ref.load(), spp_weight_map).to_dataset()
            agg_solar_ref.attrs.update({
                'units': 'dimensionless',
                'long_name': 'PVtot potential by country/region (reference)',
                'weighting': weighting_desc,
                'SOURCE': 'aggregate_epp',
                'AUTHOR': 'Colin Lenoble',
            })
            agg_solar_ref.to_netcdf(spp_ref_agg_path)

    # ------------------------------------------------------------------
    # Wind aggregation
    # ------------------------------------------------------------------
    agg_wind = xa.aggregate(wpp.load(), wpp_weight_map).to_dataset()
    agg_wind.attrs.update({
        'units': 'dimensionless',
        'long_name': 'Wind potential by country/region',
        'weighting': weighting_desc,
        'SOURCE': 'aggregate_epp',
        'AUTHOR': 'Colin Lenoble',
    })
    agg_wind.to_netcdf(
        f"{path_preprocessed}{GCM}/wpp_agg_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}_{suffix_shp}.nc")

    # Reference aggregation — v1 only
    if suffix_shp == 'v1':
        wpp_ref_agg_path = f"{path_preprocessed}{GCM}/wpp_agg_ref_{GCM}_{reanalysis}.nc"
        if not os.path.exists(wpp_ref_agg_path):
            agg_wind_ref = xa.aggregate(wpp_ref.load(), wpp_weight_map).to_dataset()
            agg_wind_ref.attrs.update({
                'units': 'dimensionless',
                'long_name': 'Wind potential by country/region (reference)',
                'weighting': weighting_desc,
                'SOURCE': 'aggregate_epp',
                'AUTHOR': 'Colin Lenoble',
            })
            agg_wind_ref.to_netcdf(wpp_ref_agg_path)


def aggregate_epp_reanalysis(
    path_preprocessed, temp_folder, shapefile_path,
    reanalysis='W5E5', suffix_shp='v1',
):
    """
    Aggregate reanalysis wpp/spp (native grid) by region.

    Reads the files produced by ``calculate_epp_reanalysis`` and applies
    the same xagg-based weighting as ``aggregate_epp``.

    suffix_shp controls the weighting scheme:
      'v1' : weighted by mean capacity factor over the full reanalysis period
      'v2' : weighted by grid-cell area only (pure geographic weighting)

    Outputs
    -------
    {path_preprocessed}/{reanalysis}/spp_agg_{reanalysis}_{suffix_shp}.nc
    {path_preprocessed}/{reanalysis}/wpp_agg_{reanalysis}_{suffix_shp}.nc
    """
    if suffix_shp not in ('v1', 'v2'):
        raise ValueError(
            f"Unknown suffix_shp {suffix_shp!r}. "
            "Expected 'v1' (capacity-factor weighted) or 'v2' (area weighted)."
        )

    wpp_path = os.path.join(path_preprocessed, reanalysis, f"wpp_day_{reanalysis}_historical_reanalysis_19790101-20191231.nc")
    spp_path = os.path.join(path_preprocessed, reanalysis, f"spp_day_{reanalysis}_historical_reanalysis_19790101-20191231.nc")

    if not os.path.exists(wpp_path) or not os.path.exists(spp_path):
        raise FileNotFoundError(
            f"Reanalysis EPP files not found in "
            f"{os.path.join(path_preprocessed, reanalysis)}. "
            "Run calculate_epp_reanalysis first."
        )

    wpp = xr.open_dataset(wpp_path)
    spp = xr.open_dataset(spp_path)
    shapefile = gpd.read_file(shapefile_path)

    wm_dir = os.path.join(temp_folder, reanalysis)
    os.makedirs(wm_dir, exist_ok=True)
    spp_wm_path = os.path.join(wm_dir, f"{reanalysis}_weightmap_spp_{suffix_shp}")
    wpp_wm_path = os.path.join(wm_dir, f"{reanalysis}_weightmap_wpp_{suffix_shp}")

    if suffix_shp == 'v1':
        if not os.path.exists(spp_wm_path):
            spp_weight_map = xa.pixel_overlaps(spp, shapefile,
                                               weights=spp.spp.mean(dim='time'))
            spp_weight_map.to_file(spp_wm_path)
        else:
            spp_weight_map = xa.read_wm(spp_wm_path)

        if not os.path.exists(wpp_wm_path):
            wpp_weight_map = xa.pixel_overlaps(wpp, shapefile,
                                               weights=wpp.wpp.mean(dim='time'))
            wpp_weight_map.to_file(wpp_wm_path)
        else:
            wpp_weight_map = xa.read_wm(wpp_wm_path)

    else:  # suffix_shp == 'v2'
        if not os.path.exists(spp_wm_path):
            spp_weight_map = xa.pixel_overlaps(spp, shapefile)
            spp_weight_map.to_file(spp_wm_path)
        else:
            spp_weight_map = xa.read_wm(spp_wm_path)

        if not os.path.exists(wpp_wm_path):
            wpp_weight_map = xa.pixel_overlaps(wpp, shapefile)
            wpp_weight_map.to_file(wpp_wm_path)
        else:
            wpp_weight_map = xa.read_wm(wpp_wm_path)

    weighting_desc = (
        'weighted by mean solar/wind capacity factor over the reanalysis period'
        if suffix_shp == 'v1' else
        'weighted by grid-cell area only (no capacity factor)'
    )

    # Solar aggregation
    agg_solar = xa.aggregate(spp.load(), spp_weight_map).to_dataset()
    agg_solar.attrs.update({
        'units': 'dimensionless',
        'long_name': 'PVtot potential by country/region',
        'weighting': weighting_desc,
        'SOURCE': 'aggregate_epp_reanalysis',
        'AUTHOR': 'Colin Lenoble',
    })
    spp_out = os.path.join(path_preprocessed, reanalysis,
                           f"spp_agg_{reanalysis}_{suffix_shp}.nc")
    agg_solar.to_netcdf(spp_out)
    print("Written aggregated spp to", spp_out)

    # Wind aggregation
    agg_wind = xa.aggregate(wpp.load(), wpp_weight_map).to_dataset()
    agg_wind.attrs.update({
        'units': 'dimensionless',
        'long_name': 'Wind potential by country/region',
        'weighting': weighting_desc,
        'SOURCE': 'aggregate_epp_reanalysis',
        'AUTHOR': 'Colin Lenoble',
    })
    wpp_out = os.path.join(path_preprocessed, reanalysis,
                           f"wpp_agg_{reanalysis}_{suffix_shp}.nc")
    agg_wind.to_netcdf(wpp_out)
    print("Written aggregated wpp to", wpp_out)


def aggregate_epp_ref_regridded(
    path_preprocessed, temp_folder, shapefile_path,
    reanalysis='W5E5', suffix_shp='v1',
):
    """
    Aggregate reanalysis wpp/spp (GCM regridded) by region.

    Reads the files produced by ``calculate_epp_reanalysis`` and applies
    the same xagg-based weighting as ``aggregate_epp``.

    suffix_shp controls the weighting scheme:
      'v1' : weighted by mean capacity factor over the full reanalysis period
      'v2' : weighted by grid-cell area only (pure geographic weighting)

    Outputs
    -------
    {path_preprocessed}/{reanalysis}/spp_agg_{reanalysis}_{suffix_shp}.nc
    {path_preprocessed}/{reanalysis}/wpp_agg_{reanalysis}_{suffix_shp}.nc
    """
    if suffix_shp not in ('v1', 'v2'):
        raise ValueError(
            f"Unknown suffix_shp {suffix_shp!r}. "
            "Expected 'v1' (capacity-factor weighted) or 'v2' (area weighted)."
        )
    
    shapefile = gpd.read_file(shapefile_path)


    wpp_paths = glob.glob(os.path.join(path_preprocessed, '*', 
                                       f"wpp_ref_*_W5E5.nc"))
    spp_paths = glob.glob(os.path.join(path_preprocessed, '*', 
                                       f"spp_ref_*_W5E5.nc"))
    
    GCM_list = [os.path.basename(p).split('_')[-2] for p in wpp_paths]

    for GCM in GCM_list:
        wpp_path = os.path.join(path_preprocessed, GCM, f"wpp_ref_{GCM}_{reanalysis}.nc")
        spp_path = os.path.join(path_preprocessed, GCM, f"spp_ref_{GCM}_{reanalysis}.nc")

        wpp_ref = xr.open_dataset(wpp_path)
        spp_ref = xr.open_dataset(spp_path)

        wm_dir = os.path.join(temp_folder, GCM)
        os.makedirs(wm_dir, exist_ok=True)
        spp_wm_path = os.path.join(wm_dir, f"{GCM}_weightmap_spp_{suffix_shp}")
        wpp_wm_path = os.path.join(wm_dir, f"{GCM}_weightmap_wpp_{suffix_shp}")

        if suffix_shp == 'v1':
            if not os.path.exists(spp_wm_path):
                spp_weight_map = xa.pixel_overlaps(spp_ref, shapefile,
                                                weights=spp_ref.sel(time=slice('1982-01-01', '2001-12-31')).spp.mean(dim='time'))
                spp_weight_map.to_file(spp_wm_path)
            else:
                spp_weight_map = xa.read_wm(spp_wm_path)

            if not os.path.exists(wpp_wm_path):
                wpp_weight_map = xa.pixel_overlaps(wpp_ref, shapefile,
                                                weights=wpp_ref.sel(time=slice('1982-01-01', '2001-12-31')).wpp.mean(dim='time'))
                wpp_weight_map.to_file(wpp_wm_path)
            else:
                wpp_weight_map = xa.read_wm(wpp_wm_path)

        else:  # suffix_shp == 'v2'
            if not os.path.exists(spp_wm_path):
                spp_weight_map = xa.pixel_overlaps(spp_ref, shapefile)
                spp_weight_map.to_file(spp_wm_path)
            else:
                spp_weight_map = xa.read_wm(spp_wm_path)

            if not os.path.exists(wpp_wm_path):
                wpp_weight_map = xa.pixel_overlaps(wpp_ref, shapefile)
                wpp_weight_map.to_file(wpp_wm_path)
            else:
                wpp_weight_map = xa.read_wm(wpp_wm_path)

        weighting_desc = (
            'weighted by mean solar/wind capacity factor over the reanalysis period'
            if suffix_shp == 'v1' else
            'weighted by grid-cell area only (no capacity factor)'
        )

        # Solar aggregation
        agg_solar = xa.aggregate(spp_ref.load(), spp_weight_map).to_dataset()
        agg_solar.attrs.update({
            'units': 'dimensionless',
            'long_name': 'PVtot potential by country/region',
            'weighting': weighting_desc,
            'SOURCE': 'aggregate_epp_reanalysis',
            'AUTHOR': 'Colin Lenoble',
        })
        spp_out = os.path.join(path_preprocessed, reanalysis,
                            f"spp_agg_ref_{GCM}_{reanalysis}_{suffix_shp}.nc")
        agg_solar.to_netcdf(spp_out)
        print("Written aggregated spp to", spp_out)

        # Wind aggregation
        agg_wind = xa.aggregate(wpp_ref.load(), wpp_weight_map).to_dataset()
        agg_wind.attrs.update({
            'units': 'dimensionless',
            'long_name': 'Wind potential by country/region',
            'weighting': weighting_desc,
            'SOURCE': 'aggregate_epp_reanalysis',
            'AUTHOR': 'Colin Lenoble',
        })
        wpp_out = os.path.join(path_preprocessed, reanalysis,
                            f"wpp_agg_ref_{GCM}_{reanalysis}_{suffix_shp}.nc")
        agg_wind.to_netcdf(wpp_out)
        print("Written aggregated wpp to", wpp_out)






def build_available_df(path_preprocessed, ssp, reanalysis='W5E5',
                       gwl_list=('GWL0-61', 'GWL1', 'GWL1-5', 'GWL2', 'GWL3')):
    """
    Scan path_preprocessed and return a DataFrame of all GCM-run pairs
    with a boolean column per GWL indicating whether the wpp_day file exists.

    Detection is based on: wpp_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.nc

    Parameters
    ----------
    path_preprocessed : str
    ssp               : str   e.g. 'ssp245'
    reanalysis        : str   e.g. 'W5E5'
    gwl_list          : sequence of GWL strings to check

    Returns
    -------
    pd.DataFrame with columns: GCM, run, ssp, <one bool col per GWL>, n_gwl_available
    """
    pattern = os.path.join(path_preprocessed, '*',
                           f"wpp_day_*_{ssp}_*_{reanalysis}.nc")
    all_files = glob.glob(pattern)

    if not all_files:
        print(f"No wpp files found under {path_preprocessed}")
        return pd.DataFrame()

    records = {}
    for fpath in all_files:
        fname = os.path.basename(fpath)
        parts = fname.replace('.nc', '').split('_')
        # Filename format: wpp_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.nc
        # Anchor on ssp and reanalysis to handle GCM names with underscores
        # e.g. EC-Earth3-Veg-LR -> parts between 'day' and ssp = GCM
        try:
            ssp_idx = parts.index(ssp)
            rea_idx = parts.index(reanalysis)
            gcm = '_'.join(parts[2:ssp_idx])
            run = parts[ssp_idx + 1]
            gwl = '_'.join(parts[ssp_idx + 2:rea_idx])
        except ValueError:
            print(f"Could not parse: {fname}, skipping.")
            continue

        key = (gcm, run)
        if key not in records:
            records[key] = {'GCM': gcm, 'run': run, 'ssp': ssp}
        records[key][gwl] = True

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(list(records.values()))

    # Ensure every expected GWL has a column (False if no file was found)
    for gwl in gwl_list:
        if gwl not in df.columns:
            df[gwl] = False
        else:
            df[gwl] = df[gwl].fillna(False)

    df['n_gwl_available'] = df[list(gwl_list)].sum(axis=1).astype(int)
    df = df[['GCM', 'run', 'ssp'] + list(gwl_list) + ['n_gwl_available']]
    df = df.sort_values(['GCM', 'run']).reset_index(drop=True)

    return df


if __name__ == "__main__":
    path_folder = '/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_raw/'
    path_preprocessed = '/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/'
    shapefile_path = '/gpfs/workdir/shared/juicce/RE_Colin/shapefile_data/shp_re.shp'
    temp_folder = '/gpfs/workdir/shared/juicce/RE_Colin/temp/'
    
    ssp = 'ssp245'
    gwl_list  = ['GWL0-61', 'GWL1', 'GWL1-5', 'GWL2', 'GWL3']
    reanalysis = 'W5E5'
    
    # --- Inventory ---
    df_available = build_available_df(path_preprocessed, ssp, reanalysis, gwl_list)
    print(df_available.to_string())
    df_to_process = df_available.copy()
    

    path_list = glob.glob(f"{path_preprocessed}*/wpp_day_*_ssp245_*_GWL0-61_W5E5.nc")
    GCM_list  = [os.path.basename(p).split('_')[-5] for p in path_list]
    run_list  = [os.path.basename(p).split('_')[-3] for p in path_list]

    

    # Load physical constants
    cfg = DEFAULT_EPP_CONFIG

    aggregate_epp_ref_regridded(path_preprocessed, temp_folder, shapefile_path, reanalysis, suffix_shp='v1')

    # unbias_GCM(GCM, run, ssp, path_preprocessed, shapefile_path,
    #            path_folder, gwl_list, reanalysis)
    # calculate_epp_reanalysis_grid_GCM(GCM, run, ssp, path_preprocessed,
    #                                    path_folder, reanalysis, shapefile_path, cfg=cfg)
    # aggregate_epp_reanalysis(path_preprocessed, temp_folder, shapefile_path,reanalysis='W5E5', suffix_shp='v1')
    # aggregate_epp_reanalysis(path_preprocessed, temp_folder, shapefile_path,reanalysis='W5E5', suffix_shp='v2')
     # --- Loop over GCM-run pairs, skip GWLs that don't exist ---
    # for _, row in df_to_process.iterrows():
    #     GCM = row['GCM']
    #     run = row['run']
    #     print(f"\n--- Processing {GCM} {run} ---")
    #     for gwl in gwl_list:
    #         if not row[gwl]:
    #             print(f"  Skipping {gwl} (file not available)")
    #             continue
    #         print(f"  Processing {gwl}")
            # calculate_epp_GCM(GCM, run, ssp, path_preprocessed, gwl, cfg=cfg)
            #aggregate_epp(GCM, run, ssp, path_preprocessed, temp_folder,
            #              gwl, shapefile_path, reanalysis, suffix_shp='v1')