import xarray as xr
import xagg as xa
import pandas as pd
import numpy as np
import gcsfs
import os
import warnings
from tqdm import tqdm

# ---- Configuration ----
OUTPUT_DIR = './data/raw/'

# Bounding box from INDIA_STATES.geojson
INDIA_BBOX = {'lat': [6.75, 37.08], 'lon': [68.09, 97.42]}

EXPERIMENTS  = ['historical', 'ssp119', 'ssp245', 'ssp370', 'ssp585']
VARIABLES    = ['rsds', 'uas', 'vas', 'tas', 'tasmax']
TABLE_ID     = 'day'
MAX_RUNS     = 1      # max ensemble members per (model, experiment)
OVERWRITE    = False


def build_query():
    exp_str = ' or '.join([f"experiment_id == '{e}'" for e in EXPERIMENTS])
    var_str = ' or '.join([f"variable_id == '{v}'" for v in VARIABLES])
    return f"table_id == '{TABLE_ID}' and ({exp_str}) and ({var_str})"


def spatial_subset(ds):
    """Clip dataset to India bbox, handling both ascending and descending lat."""
    lat0, lat1 = INDIA_BBOX['lat']
    lon0, lon1 = INDIA_BBOX['lon']

    # some models store lat descending
    if float(ds.lat[0]) > float(ds.lat[-1]):
        lat_slice = slice(lat1, lat0)
    else:
        lat_slice = slice(lat0, lat1)

    return ds.sel(lat=lat_slice, lon=slice(lon0, lon1))


def main(gcm_list):
    fs = gcsfs.GCSFileSystem(token='anon', access='read_only')

    print('Fetching CMIP6 catalogue...')
    cmip6_datasets = pd.read_csv(
        'https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv'
    )
    cmip6_sub = cmip6_datasets.query(build_query())
    cmip6_sub = cmip6_sub[cmip6_sub['source_id'].isin(gcm_list)].copy()

    if len(cmip6_sub) == 0:
        warnings.warn('No matching files found in pangeo for the requested models.')
        return

    # One grid type per (model, member, experiment, variable)
    cmip6_sub = (
        cmip6_sub
        .sort_values('grid_label')
        .groupby(['source_id', 'member_id', 'experiment_id', 'variable_id'])
        .first()
        .reset_index()
    )

    # Limit ensemble members per (model, experiment) — pick the same member across variables
    if MAX_RUNS is not None:
        # Identify the first MAX_RUNS members per (model, experiment)
        kept_members = (
            cmip6_sub[['source_id', 'experiment_id', 'member_id']]
            .drop_duplicates()
            .groupby(['source_id', 'experiment_id'])
            .head(MAX_RUNS)
        )
        cmip6_sub = cmip6_sub.merge(kept_members, on=['source_id', 'experiment_id', 'member_id'])

    print(f'Processing {len(cmip6_sub)} (model × member × experiment × variable) combinations.')

    for _, row in tqdm(cmip6_sub.iterrows(), total=len(cmip6_sub)):
        mod    = row['source_id']
        member = row['member_id']
        exp    = row['experiment_id']
        var    = row['variable_id']

        out_dir = os.path.join(OUTPUT_DIR, mod)
        os.makedirs(out_dir, exist_ok=True)

        # Skip if a file for this combo already exists
        prefix = f"{var}_{TABLE_ID}_{mod}_{exp}_{member}_"
        if not OVERWRITE and any(f.startswith(prefix) for f in os.listdir(out_dir)):
            print(f'Already exists: {prefix}*, skipping.')
            continue

        try:
            print(f'\nLoading {mod} | {member} | {exp} | {var}...')
            ds = xr.open_zarr(fs.get_mapper(row['zstore']), consolidated=True)

            # Drop duplicate time steps (known issue in some models)
            if len(np.unique(ds.time)) < ds.sizes['time']:
                ds = ds.drop_duplicates('time')
                warnings.warn(f'{mod} {member} {exp} {var}: dropped duplicate time steps.')

            # Standardise coordinate names and lon range to -180:180
            ds = xa.fix_ds(ds)

            # Clip to India
            ds = spatial_subset(ds)

            if ds.sizes.get('lat', 0) == 0 or ds.sizes.get('lon', 0) == 0:
                print(f'  Empty spatial extent after clipping, skipping.')
                continue

            # Build filename from actual time extent in the file
            t0 = str(ds.time.values[0])[:7].replace('-', '')
            t1 = str(ds.time.values[-1])[:7].replace('-', '')
            out_fn = os.path.join(out_dir, f"{prefix}{t0}-{t1}_india.nc")

            ds.attrs['SOURCE']      = 'download_cmip6_india.py'
            ds.attrs['DESCRIPTION'] = f'Full {exp} run clipped to India bbox {INDIA_BBOX}'

            ds = ds.load()
            ds.to_netcdf(out_fn)
            print(f'  Saved -> {out_fn}')

        except Exception as e:
            print(f'  Error with {mod} {member} {exp} {var}: {e}')
            continue


if __name__ == '__main__':
    gcm_list = ['CanESM5']
    main(gcm_list)
