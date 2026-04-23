import xarray as xr
import numpy as np
from pathlib import Path
import gc
from dataclasses import dataclass

# -------------------------
# EPP physical constants
# -------------------------

@dataclass
class EPPConfig:
    """
    Physical constants for wind and solar capacity factor calculations.

    Wind power curve
    ----------------
    vr  : rated wind speed (m/s)
    vci : cut-in wind speed (m/s)
    vco : cut-out wind speed (m/s)
    wind_height_exponent : Hellmann exponent for 10 m -> 80 m extrapolation

    Solar PV cell temperature model (Huld et al.)
    ---------------------------------------------
    gamma : temperature coefficient of PV efficiency (K^-1)
    T_ref : reference cell temperature (°C)
    G_stc : irradiance at standard test conditions (W m^-2)
    c_1..c_4 : NOCT-based cell temperature model coefficients
    """
    vr:  float = 13.0
    vci: float = 3.5
    vco: float = 25.0
    wind_height_exponent: float = 0.143

    gamma: float = -0.005
    T_ref: float = 25.0
    G_stc: float = 1000.0
    c_1: float =  4.3
    c_2: float =  0.943
    c_3: float =  0.028
    c_4: float = -1.528


DEFAULT_EPP_CONFIG = EPPConfig()


# -------------------------
# ERA5 loading helpers
# -------------------------

def _load_instant_var(grib_path: str, short_name: str, vname: str) -> xr.DataArray:
    """Load a single instantaneous ERA5 variable (u10, v10, t2m) from GRIB."""
    ds = xr.open_dataset(
        grib_path,
        engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'shortName': short_name}},
        chunks={'time': 24},
    )
    actual = [v for v in ds.data_vars if v != 'valid_time']
    if not actual:
        raise KeyError(f"Variable {short_name!r} not found in GRIB file")
    da = ds[actual[0]].rename(vname)
    keep = {'time', 'latitude', 'longitude', 'lat', 'lon'}
    da = da.drop_vars([c for c in da.coords if c not in keep], errors='ignore')
    return da


def _load_ssrd(grib_path: str) -> xr.DataArray:
    """
    Load ERA5 ssrd from GRIB and return hourly W/m².

    cfgrib exposes ssrd as (time=N_inits, step=12, lat, lon).
    Both forecast init types (06 UTC and 18 UTC) store PER-STEP hourly J/m²
    values — NOT cumulative from init. The 18 UTC values increase monotonically
    because the valid_times span nighttime→morning (solar naturally ramps up);
    the 06 UTC values decrease because the valid_times span noon→night.
    No deaccumulation (differencing) is needed — just divide by 3600.
    """
    ds = xr.open_dataset(
        grib_path,
        engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'shortName': 'ssrd'}},
    )
    ssrd = ds['ssrd'].load()              # (time=N_inits, step=12, lat, lon)
    valid_time = ds['valid_time'].load()  # (time=N_inits, step=12)

    lat_name = 'latitude' if 'latitude' in ssrd.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in ssrd.dims else 'lon'

    # Per-step hourly J/m² → W/m²; NaN = no radiation (clip negatives from encoding)
    hourly = np.clip(ssrd.fillna(0.0).values / 3600.0, 0.0, None)

    n_init, n_step, n_lat, n_lon = hourly.shape
    flat_time = valid_time.values.flatten()
    flat_data = hourly.reshape(n_init * n_step, n_lat, n_lon)

    # Sort chronologically and drop any duplicate timestamps at init boundaries
    sort_idx = np.argsort(flat_time)
    flat_time = flat_time[sort_idx]
    flat_data = flat_data[sort_idx]
    _, unique_idx = np.unique(flat_time, return_index=True)
    flat_time = flat_time[unique_idx]
    flat_data = flat_data[unique_idx]

    return xr.DataArray(
        flat_data,
        dims=['time', lat_name, lon_name],
        coords={
            'time': flat_time,
            lat_name: ssrd[lat_name].values,
            lon_name: ssrd[lon_name].values,
        },
        name='ssrd',
    )


def load_era5_grib(grib_path: str) -> xr.Dataset:
    """
    Load ERA5 u10, v10, t2m and deaccumulated ssrd from a GRIB file.

    Returns a merged Dataset on a common hourly time axis with
    dimensions (time, lat, lon).
    """
    grib_path = str(grib_path)
    print("  Loading u10, v10, t2m...")
    arrays = {
        'u10':  _load_instant_var(grib_path, '10u', 'u10'),
        'v10':  _load_instant_var(grib_path, '10v', 'v10'),
        't2m':  _load_instant_var(grib_path, '2t',  't2m'),
    }
    print("  Loading and deaccumulating ssrd...")
    arrays['ssrd'] = _load_ssrd(grib_path)

    ds = xr.merge(list(arrays.values()), compat='override', join='inner')

    # Rename latitude/longitude -> lat/lon
    rename_map = {}
    if 'latitude' in ds.dims:
        rename_map['latitude'] = 'lat'
    if 'longitude' in ds.dims:
        rename_map['longitude'] = 'lon'
    if rename_map:
        ds = ds.rename(rename_map)

    # Sort coordinates (index arrays only — no data load)
    lat_idx  = np.argsort(ds.lat.values)
    lon_idx  = np.argsort(ds.lon.values)
    time_idx = np.argsort(ds.time.values)
    ds = ds.isel(lat=lat_idx, lon=lon_idx, time=time_idx)

    return ds


# -------------------------
# Capacity factor calculations
# -------------------------

def compute_scf(t2m_c: xr.DataArray, ssrd_wm2: xr.DataArray,
                sfcwind: xr.DataArray, cfg: EPPConfig) -> xr.DataArray:
    """
    Solar capacity factor (dimensionless, 0-1+) using the Huld cell
    temperature model.

    ERA5 provides hourly t2m, so we use it directly — no tasmax averaging.
    """
    T_cell = (cfg.c_1
              + cfg.c_2 * t2m_c
              + cfg.c_3 * ssrd_wm2
              + cfg.c_4 * sfcwind)
    P_R = 1.0 + cfg.gamma * (T_cell - cfg.T_ref)
    scf = P_R * (ssrd_wm2 / cfg.G_stc)
    return scf


def compute_wcf(sfcwind_10m: xr.DataArray, cfg: EPPConfig) -> xr.DataArray:
    """
    Wind capacity factor (dimensionless, 0-1) using a standard power curve
    with wind speed extrapolated from 10 m to 80 m via the Hellmann law.
    """
    wind_80m = sfcwind_10m * (80.0 / 10.0) ** cfg.wind_height_exponent

    wcf = xr.where(wind_80m < cfg.vci, 0.0, wind_80m)
    wcf = xr.where(wcf >= cfg.vco, 0.0, wcf)
    wcf = xr.where((wcf >= cfg.vr) & (wcf < cfg.vco), 1.0, wcf)
    wcf = xr.where(
        (wcf >= cfg.vci) & (wcf < cfg.vr),
        (wcf**3 - cfg.vci**3) / (cfg.vr**3 - cfg.vci**3),
        wcf,
    )
    return wcf


# -------------------------
# Main function
# -------------------------

def calculate_epp_era5(
    grib_path: str,
    output_dir: str = "data/proc/era5",
    cfg: EPPConfig = DEFAULT_EPP_CONFIG,
    chunks: dict = None,
) -> tuple[str, str]:
    """
    Compute hourly wind (wcf) and solar (scf) capacity factors from an
    ERA5 GRIB file and write them to NetCDF.

    Parameters
    ----------
    grib_path  : path to the ERA5 .grib file
    output_dir : directory where wcf/scf NetCDF files are written
    cfg        : EPPConfig with physical constants
    chunks     : dask chunk dict (default: {'time': 24, 'lat': -1, 'lon': -1})

    Returns
    -------
    (path_wcf, path_scf) as strings
    """
    if chunks is None:
        chunks = {'time': 24, 'lat': -1, 'lon': -1}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(grib_path).stem
    path_wcf = str(output_dir / f"wcf_{stem}.nc")
    path_scf = str(output_dir / f"scf_{stem}.nc")

    print(f"Loading ERA5 GRIB: {grib_path}")
    ds = load_era5_grib(grib_path)
    ds = ds.chunk(chunks)

    # Wind speed at 10 m
    sfcwind = np.hypot(ds['u10'], ds['v10'])

    # ssrd already deaccumulated to W/m² by load_era5_grib
    ssrd = ds['ssrd']

    # Temperature: K -> °C
    t2m_c = ds['t2m'] - 273.15

    # --- Wind capacity factor ---
    print("Computing wind capacity factor (wcf)...")
    wcf = compute_wcf(sfcwind, cfg)
    ds_wcf = wcf.to_dataset(name='wcf')
    ds_wcf['wcf'] = ds_wcf['wcf'].astype('f4')
    ds_wcf.attrs.update({
        'description': 'Hourly wind capacity factor from ERA5',
        'units': 'dimensionless',
        'long_name': 'Wind capacity factor',
        'source': 'calculate_epp_era5.py',
        'author': 'Colin Lenoble',
        'wind_height_m': 80,
        'vci': cfg.vci,
        'vr': cfg.vr,
        'vco': cfg.vco,
    })
    ds_wcf = ds_wcf.compute()
    ds_wcf.to_netcdf(path_wcf, mode='w')
    print(f"Written wcf to {path_wcf}")
    ds_wcf.close()

    # --- Solar capacity factor ---
    print("Computing solar capacity factor (scf)...")
    scf = compute_scf(t2m_c, ssrd, sfcwind, cfg).clip(min=0.0, max=1.0)
    ds_scf = scf.to_dataset(name='scf')
    ds_scf['scf'] = ds_scf['scf'].astype('f4')
    ds_scf.attrs.update({
        'description': 'Hourly solar capacity factor from ERA5',
        'units': 'dimensionless',
        'long_name': 'Solar (PV) capacity factor',
        'source': 'calculate_epp_era5.py',
        'author': 'Colin Lenoble',
        'note': 'Uses hourly t2m directly (no tasmax averaging needed)',
        'gamma': cfg.gamma,
        'T_ref': cfg.T_ref,
        'G_stc': cfg.G_stc,
    })
    ds_scf = ds_scf.compute()
    ds_scf.to_netcdf(path_scf, mode='w')
    print(f"Written scf to {path_scf}")
    ds_scf.close()

    ds.close()
    gc.collect()
    print("Done.")
    return path_wcf, path_scf


# -------------------------
# Entry point
# -------------------------

if __name__ == "__main__":
    GRIB_PATH = "data/raw/era5/era5_india_2025_hourly.grib"
    OUTPUT_DIR = "data/proc/era5"

    calculate_epp_era5(
        grib_path=GRIB_PATH,
        output_dir=OUTPUT_DIR,
        cfg=DEFAULT_EPP_CONFIG,
    )