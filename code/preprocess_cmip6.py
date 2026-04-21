import xarray as xr
import xagg as xa
import pandas as pd
import numpy as np
import cftime
from tqdm.notebook import tqdm
import re
from operator import itemgetter # For list subsetting but this is idiotic
import intake
import gcsfs
import os
import warnings 

from funcs_support import (get_varlist,get_params)
dir_list = get_params()


def main(gcm_list):
        # Don't donwload if more than 10 runs already downloaded
    max_runs = 10
    gwl_info = pd.read_csv(dir_list['aux']+'mathause-cmip_warming_levels-f47853e/warming_levels/cmip6_all_ens/csv/cmip6_warming_levels_all_ens_1850_1900_grid.csv',
                        skiprows=4,skipinitialspace=True)
    gwl_info.columns = [t.strip() for t in gwl_info.columns]
    subset_params_all = [{'fn_suffix':'','lat':[-58,73],'lon':[-180,180]}]

    # data_params_all = [[{'gwl':gwl,'experiment_id':exp,'table_id':'day','variable_id':'tasmax'}
    #                      for gwl in [0.61,1.5,2,3,4]] for exp in ['ssp245']] #[,1.5,2,3,4]#['ssp119','ssp245','ssp370','ssp585']
    data_params_all = [[{'gwl':gwl,'experiment_id':'ssp245','table_id':'day','variable_id':var}
                        for var in ['rsds','uas','vas','tas','tasmax']] for gwl in [0.61,1.5,2,3,4]]

    data_params_all = [item for sublist in data_params_all for item in sublist]
    ## Prepare the full query for all the datasets that will end up getting use in this 
    # process - this is to create the master dataset, so to build up the 'model' and 
    # 'experiment' dimension in the dataset with all the values that will end up used
    data_params_esgf = {k:v for k,v in data_params_all[0].items() if k != 'gwl'}

    source_calls = np.zeros(len(data_params_esgf.keys()))

    for key in data_params_esgf.keys():
        if len(np.unique([x[key] for x in data_params_all]))==1:
            source_calls[list(data_params_esgf.keys()).index(key)] = 1
            

    # First get all the ones with the same value for each key 
    subset_query = ' and '.join([k+" == '"+data_params_esgf[k]+"'" for k in itemgetter(*source_calls.nonzero()[0])(list(data_params_esgf.keys())) if k != 'other'])

    # Now add all that are different between subset params - i.e. those that need an OR statement
    # These have to be in two statements, because if there's only one OR'ed statement, then the 
    # for k in statement goes through the letters instead of the keys. 
    if len((source_calls-1).nonzero()[0])==1:
        subset_query=subset_query+' and ('+') and ('.join([' or '.join([k+" == '"+data_params[k]+"'" for data_params in data_params_all]) 
                for k in [itemgetter(*(source_calls-1).nonzero()[0])(list(data_params_esgf.keys()))] if k != 'other'])+')'
    elif len((source_calls-1).nonzero()[0])>1:
        subset_query=subset_query+' and ('+') and ('.join([' or '.join([k+" == '"+data_params[k]+"'" for data_params in data_params_all]) 
                for k in itemgetter(*(source_calls-1).nonzero()[0])(list(data_params_esgf.keys())) if k != 'other'])+')'

    # Add historical to search, since some GWLs duck into the historical period... 
    subset_query = re.sub(r"experiment_id == '"+data_params_esgf['experiment_id']+"'",
                        r"(experiment_id == '"+data_params_esgf['experiment_id']+"' or experiment_id == 'historical')",
                        subset_query)
    # Access google cloud storage links
    fs = gcsfs.GCSFileSystem(token='anon', access='read_only')
    # Get info about CMIP6 datasets
    
    #gcm_list = ['NorESM2-MM']

    cmip6_datasets = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
    # Get subset based on the data params above (for all search parameters)
    cmip6_sub = cmip6_datasets.query(subset_query)

    #cmip6_sub = cmip6_sub[cmip6_sub['source_id'].isin(gcm_list)]

    if len(cmip6_sub) == 0:
        warnings.warn('Query unsuccessful, no matching files found in pangeo! Check to make sure your table_id matches the domain - SSTs are listed as "Oday" instead of "day" for example')
        
    #print the gcm where you don't have the 5 variables
    gcm_missing = []
    for gcm in gcm_list:
        if len(cmip6_sub[cmip6_sub['source_id']==gcm]['variable_id'].unique()) != 5:
            print(gcm)
            print(cmip6_sub[cmip6_sub['source_id']==gcm]['variable_id'].unique())
            gcm_missing.append(gcm)

    # Get the list of all the datasets that are missing
    cmip6_sub = cmip6_sub[cmip6_sub['source_id'].isin(gcm_list)]
    cmip6_sub = cmip6_sub[~cmip6_sub['source_id'].isin(gcm_missing)]
    cmip6_sub.source_id.unique().shape,len(gcm_list)
    #keep only the rows where tehre are ²

    cmip6_sub_sub = cmip6_sub.groupby(['source_id','activity_id','experiment_id','table_id','member_id']).size().reset_index().rename(columns={0:'count'}).query('count == 5')
    #extract from the cmip6_sub the rows that are in cmip6_sub_sub
    cmip6_final = cmip6_sub.merge(cmip6_sub_sub, on=['source_id','activity_id','experiment_id','table_id','member_id'], how='inner')
    cmip6_final = cmip6_final.drop(columns='count')
    cmip6_sub = cmip6_final

    data_params = data_params_all[0]
    # Get GWL information
    gwl_info_sub = gwl_info.loc[gwl_info.warming_level == data_params['gwl'],:]
    # Match pangeo column names
    gwl_info_sub.columns = ['source_id','member_id','experiment_id','grid_label','warming_level','start_year','end_year']
    # I hope that the grid labels aren't needed here, but drop_duplicates gives the same number
    # so I think they just did one grid type per GWL
    gwl_info_sub = gwl_info_sub.drop(['grid_label','warming_level'],axis=1)

    # Get cmip6 files with desired experiment
    process_list = cmip6_sub.loc[cmip6_sub['experiment_id'] == data_params['experiment_id'],:]

    # Figure out which files both exist on the pangeo data store but are also in the GWL archive
    process_list = (process_list.set_index(['source_id', 'experiment_id', 'member_id']).
                    join(gwl_info_sub.set_index(['source_id', 'experiment_id', 'member_id']),
                    how='inner')).reset_index()

    # Just get one grid type from each model
    process_list = process_list.groupby(['source_id','experiment_id','member_id']).apply(lambda x: x.sort_values(by='grid_label').iloc[0,:],
                                                                    include_groups=False).reset_index()

    #------ Process by variable and dataset in the subset ------

    overwrite=False

    # Overwrites / reloads any files that contain both historical and future experiment 
    # data - this is to ensure files are unaffected by a loading bug in those specific
    # situations, where it was possible for values > 2014 to be from a different experiment
    # than listed on the filename
    redo_hist_overlap_runs = False

    # Flag to not download any "new" models 
    # (e.g., I deleted EC-Earth3 to make space, don't want to re-download it here)
    subset_to_existing_mods = True

    for data_params in data_params_all:
        # Get GWL information
        gwl_info_sub = gwl_info.loc[gwl_info.warming_level == data_params['gwl'],:]
        # Match pangeo column names
        gwl_info_sub.columns = ['source_id','member_id','experiment_id','grid_label','warming_level','start_year','end_year']
        # I hope that the grid labels aren't needed here, but drop_duplicates gives the same number
        # so I think they just did one grid type per GWL
        gwl_info_sub = gwl_info_sub.drop(['grid_label','warming_level'],axis=1)

        # Get cmip6 files with desired experiment
        #process_list = cmip6_sub.loc[cmip6_sub['experiment_id'] == data_params['experiment_id'],:]
        process_list = cmip6_sub.loc[[k in [data_params['experiment_id']] for k in cmip6_sub['experiment_id']],:]
            
        # Figure out which files both exist on the pangeo data store but are also in the GWL archive
        process_list = (process_list.set_index(['source_id', 'experiment_id', 'member_id']).
                        join(gwl_info_sub.set_index(['source_id', 'experiment_id', 'member_id']),
                        how='inner')).reset_index()
        
        # Add in historical rows
        hist_rows = []
        for r in process_list.iterrows():
            hist_rows.append(cmip6_sub.loc[pd.concat([(cmip6_sub['experiment_id'] == 'historical'),*[cmip6_sub[k] == r[1][k] for k in ['source_id','member_id','variable_id']]],axis=1).all(axis=1),:])
        hist_rows = pd.concat(hist_rows)
            
        # Just get one grid type from each model
        process_list = process_list.groupby(['source_id','experiment_id','member_id', 'variable_id']).apply(lambda x: x.sort_values(by='grid_label').iloc[0,:],
                                                                        include_groups=False).reset_index()
        
        # If only downloading data for models for which other data has already been downloaded...
        if subset_to_existing_mods:
            process_list = process_list.iloc[[os.path.exists(dir_list['raw']+row[1]['source_id']) for row in process_list.iterrows()],:]    
    
        # Limit to just a max number of runs per model, if desired
        if max_runs is not None:
            process_list = process_list.groupby(['source_id','experiment_id']).head(max_runs)
        # 
        if len(process_list) == 0:
            warnings.warn('Query unsuccessful, no files present in both the pangeo datastore and the GWL list for '+
                        '; '.join([k+': '+str(v) for k,v in data_params.items()])+'! Check your query and try again.')
            continue

        for row in tqdm([row for row in process_list.iterrows()]):
            member_id = row[1].member_id
            mod = row[1].source_id
            print('processing '+mod+' '+member_id+'!')
            
            if row[1]['start_year'] < 2015:
                # If start_year before 2015, then historical data is needed.
                # Since cmip6_sub has only the desired experiment + historical,
                # it suffices to just find matching source/member/variable_ids
                # in the cmip6_sub file. 
                #load_files = process_list.loc[pd.concat([process_list[k] == row[1][k] for k in ['source_id','member_id','variable_id']],axis=1).all(axis=1),:]
                load_files = pd.concat([hist_rows.loc[pd.concat([hist_rows[k] == row[1][k] for k in ['source_id','member_id','variable_id','grid_label']],axis=1).all(axis=1),:],
                            row[1].to_frame().T])
                
                if len(load_files) < 2:
                    warnings.warn('No historical file found for '+', '.join([row[1][k] for k in ['source_id','member_id','variable_id']])+' despite start year '+
                                str(row[1]['start_year'])+'; skipped.')
                    continue
            else:
                load_files = row[1].to_frame().T

            try:
                # Set output filenames
                output_fns = [None]*len(subset_params_all)
                path_exists = [None]*len(subset_params_all)

                # Process by different subset parameters
                for subset_params in subset_params_all:
                    #-------- Open --------
                    # Open dataset(s)
                    ds = [xr.open_zarr(fs.get_mapper(url),consolidated=True)
                        for url in load_files.zstore]
                    
                    # Get time subset
                    try:
                        time_subset = slice(str(row[1]['start_year'])+'-01-01',
                                str(row[1]['end_year'])+'-12-'+str(ds[0].time.dt.daysinmonth.max().values))
                    except:
                        print('issue with time subset; skipping')
                        
                    
                    for ds_idx in range(len(ds)):
                        if len(np.unique(ds[ds_idx].time)) < ds[ds_idx].sizes['time']:
                            ds[ds_idx] = ds[ds_idx].drop_duplicates('time')
                            warnings.warn('Had to drop duplicate time coordinates with '+', '.join([row[1][k] for k in ['source_id','member_id','variable_id']]))
                    
                    # Concatenate along time 
                    try:
                        ds = xr.concat([d.sel(time=time_subset) for d in ds],
                                        dim='time')        
                    except:
                        warnings.warn('Issue with selecting times in '+', '.join([row[1][k] for k in ['source_id','member_id','variable_id']])+', skipping for now.')
                        # Not continue since this is literally still just the filename check
                        raise Exception
                    #-------- Filename check --------
                    # Get time string for filename
                    time_str = re.sub(r'-','',time_subset.start)+'-'+re.sub(r'-','',time_subset.stop)
                    
                    # Get filename
                    output_fns[subset_params_all.index(subset_params)] = (dir_list['raw']+row[1]['source_id']+'/'+
                                                                        data_params['variable_id']+'_'+
                                                                        data_params['table_id']+'_'+row[1]['source_id']+'_'+
                                                                        data_params['experiment_id']+'_'+row[1]['member_id']+'_'+
                                                                        time_str+'_'+
                                                                        'GWL'+re.sub(r'\.','-',str(data_params['gwl']))+
                                                                        subset_params['fn_suffix']+'.nc')

                    # Add plev_subset field to varname, if necessary
                    if 'other' in data_params.keys(): 
                            if 'plev_subset' in data_params['other'].keys():
                                output_fns[subset_params_all.index(subset_params)] = re.sub(data_params['variable_id'],
                                                                                        data_params['other']['plev_subset']['outputfn'],
                                                                                    output_fns[subset_params_all.index(subset_params)])

                    # Figure out if path exists
                    path_exists[subset_params_all.index(subset_params)] = os.path.exists(output_fns[subset_params_all.index(subset_params)])

                # Filter if files already exist

                # import pdb; pdb.set_trace()
                if (not overwrite) and (all(path_exists) and (not (redo_hist_overlap_runs and (row[1]['start_year']<2015) and (row[1]['end_year']>2014)))):
                    print('All files already created for '+data_params['variable_id']+' '+
                                                                        data_params['table_id']+' '+mod+' '+
                                                                        data_params['experiment_id']+' '+member_id+
                        ', '+str(row[1]['start_year'])+'-'+str(row[1]['end_year'])+', skipped.')
                    continue
                elif any(path_exists):
                    if (overwrite) or (redo_hist_overlap_runs and (row[1]['start_year']<2015) and (row[1]['end_year']>2014)):
                        for subset_params in subset_params_all:
                            if path_exists[subset_params_all.index(subset_params)]:
                                os.remove(output_fns[subset_params_all.index(subset_params)])
                                warnings.warn('All files already exist for '+data_params['variable_id']+' '+
                                                                                    data_params['table_id']+' '+mod+' '+
                                                                                    data_params['experiment_id']+' '+member_id+
                                            ', '+str(row[1]['start_year'])+'-'+str(row[1]['end_year'])+
                                            ', because OVERWRITE=TRUE these files have been deleted.')

                
                #-------- Some housecleaning on the whole file --------
                # Rename to lat/lon, reindex to -180:180 lon
                ds = xa.fix_ds(ds)

                # Fix coordinate doubling (this was an issue in NorCPM1, 
                # where thankfully the values of the variables were nans,
                # though I still don't know how this happened - some lat
                # values were doubled within floating point errors)
                if 'lat' in ds[data_params['variable_id']].sizes:
                    if len(np.unique(np.round(ds.lat.values,10))) != ds.sizes['lat']:
                        ds = ds.isel(lat=(~np.isnan(ds.isel(lon=1,time=1)[data_params['variable_id']].values)).nonzero()[0],drop=True)
                        warnings.warn('Model '+ds.source_id+' has duplicate lat values; attempting to compensate by dropping lat values that are nan in the main variable in the first timestep')
                    if len(np.unique(np.round(ds.lon.values,10))) != ds.sizes['lon']:
                        ds = ds.isel(lon=(~np.isnan(ds.isel(lat=1,time=1)[data_params['variable_id']].values)).nonzero()[0],drop=True)
                        warnings.warn('Model '+ds.source_id+' has duplicate lon values; attempting to compensate by dropping lon values that are nan in the main variable in the first timestep')

                # Drop time duplicates
                ds = ds.drop_duplicates('time')
                
                # Sort by time, if not sorted (this happened with
                # a model; keeping a warning, cuz this seems weird)
                if 'time' in subset_params:
                    if (ds.time.values != np.sort(ds.time)).any():
                        warnings.warn('Model '+ds.source_id+' has an unsorted time dimension.')
                        ds = ds.sortby('time')

                #-------- Process by subset --------
                # Now, save by the subsets desired in subset_params_all above
                for subset_params in subset_params_all:
                    # Make sure this file hasn't already been processed
                    if (not overwrite) and os.path.exists(output_fns[subset_params_all.index(subset_params)]):
                        warnings.warn(output_fns[subset_params_all.index(subset_params)]+' already exists; skipped.')
                        continue

                    # Make sure the target directory exists
                    if not os.path.exists(dir_list['raw']+row[1]['source_id']+'/'):
                        os.mkdir(dir_list['raw']+row[1]['source_id']+'/')
                        warnings.warn('Directory '+dir_list['raw']+row[1]['source_id']+'/'+' created!')

                    # Create copy, for this subset 
                    ds_tmp = ds.copy()
                    
                    # Subset by space as set in subset_params
                    if 'lat' in subset_params.keys():
                        if not 'lat' in ds[data_params['variable_id']].sizes:
                            ds_tmp = ds_tmp.where((ds_tmp.lat >= subset_params['lat'][0]) & (ds_tmp.lat <= subset_params['lat'][1]) &
                            (ds_tmp.lon >= subset_params['lon'][0]) & (ds_tmp.lon <= subset_params['lon'][1]),drop=True)
                        else:
                            ds_tmp = (ds_tmp.sel(lat=slice(*subset_params['lat']),
                                                lon=slice(*subset_params['lon'])))

                    # If subsetting by pressure level...
                    if 'other' in data_params.keys():
                        if 'plev_subset' in data_params['other'].keys():
                            # Have to use np.allclose for floating point errors
                            try:
                                ds_tmp = ds_tmp.isel(plev=np.where([np.allclose(p,data_params['other']['plev_subset']['plev']) for p in ds_tmp.plev])[0][0])
                                ds_tmp = ds_tmp.rename({data_params['variable_id']:data_params['other']['plev_subset']['outputfn']})
                            except KeyError:
                                print('The pressure levels: ')
                                print(ds_tmp.plev.values)
                                print(' do not contain '+str(data_params['other']['plev_subset']['plev'])+'; skipping.')
                                del ds_tmp
                                continue
                                
                    # Put in attributes
                    ds_tmp.attrs['SOURCE'] = 'preprocess_cmip6_bygwl.ipynb'
                    ds_tmp.attrs['DESCRIPTION'] = '20 years of data surrounding the central year of GWL'+str(data_params['gwl'])
                    if row[1]['start_year']<2015:
                        ds_tmp.attrs['DESCRIPTION2'] = 'All data from before 2015 is from the "historical" run with the same member id.'
                    ds_tmp.attrs['GWL_SOURCE'] = 'GWLs are from Hauser et al., 2022, https://zenodo.org/records/7390473'
                    
                    
                    # Save as NetCDF file
                    if ds_tmp.sizes['time']>0:
                        try:
                            ds_tmp = ds_tmp.load()
                            
                            sizes_0 = [dim for dim,siz in ds_tmp[data_params['variable_id']].sizes.items() if siz==0]
                            if len(sizes_0)>0:
                                
                                print('issue with file '+output_fns[subset_params_all.index(subset_params)]+
                                    ', dims '+', '.join(sizes_0)+' are size 0; skipping')
                                continue
                            
                            ds_tmp.to_netcdf(output_fns[subset_params_all.index(subset_params)])
                        except ValueError:
                            print('issue with export; skipping')
                            #del ds_tmp
                            continue
                    else:
                        print('time dimension is 0, skipping')
    #                    raise Error
                        continue
                        

                    # Status update
                    print(output_fns[subset_params_all.index(subset_params)]+' processed!')

                del ds 
                try:
                    del ds_tmp
                except:
                    pass
                del subset_params
            except AssertionError:
                print('checksum error with model '+row[1]['source_id']+', skipping for now.')
                continue
            except:
                print('issue with model '+row[1]['source_id']+', skipping for now.')



if __name__ == '__main__':
    gcm_list = ['ACCESS-CM2', 'ACCESS-ESM1-5','AWI-CM-1-1-MR','BCC-CSM2-MR','CAMS-CSM1-0','CanESM5','CMCC-CM2-SR5','CNRM-CM6-1', 'CNRM-ESM2-1',
                'EC-Earth3-Veg-LR','GFDL-CM4',
                'GFDL-ESM4','HadGEM3-GC31-LL','IITM-ESM','INM-CM5-0','IPSL-CM6A-LR', 'KACE-1-0-G','MIROC6','MIROC-ES2L',
                'MPI-ESM1-2-LR','MRI-ESM2-0','NESM3','NorESM2-MM','TaiESM1', 'UKESM1-0-LL']
    #gcm_list = ['UKESM1-0-LL']
    main(gcm_list)