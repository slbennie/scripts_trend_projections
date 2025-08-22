#!/usr/bin/env python
# coding: utf-8

# In[3]:


#getting the correct calendar (model dependent.)
def get_time_bounds(calendar_type, start, end):
    import xarray as xr
    import cftime
    from datetime import datetime
    #1850-2015 all of 2014 - none of 2015.
    if calendar_type == cftime.DatetimeNoLeap:
        return cftime.DatetimeNoLeap(start,1,16), cftime.DatetimeNoLeap(end,1,16)
    elif calendar_type == cftime.Datetime360Day:
        return cftime.Datetime360Day(start,1,16), cftime.Datetime360Day(end-1,12,16)
    else:
        return datetime(start,1,16), datetime(end,1,16)


# In[2]:


#finding all the models that have ensembles for that experiment.
def get_models_for_experiment(experiment):
    if experiment == 'historical':
        model = ['ACCESS-ESM1-5','CanESM5','CMCC-CM2-SR5','FGOALS-g3','GISS-E2-1-G','HadGEM3-GC31-LL','IPSL-CM6A-LR','MIROC6','MPI-ESM1-2-LR','NorESM2-LM']
    elif experiment == 'hist-aer':
        model = ['ACCESS-ESM1-5','CanESM5','CMCC-CM2-SR5','CNRM-CM6-1','FGOALS-g3','GISS-E2-1-G','HadGEM3-GC31-LL','IPSL-CM6A-LR','MIROC6','MPI-ESM1-2-LR','NorESM2-LM']
    elif experiment == 'hist-GHG':
        model = ['ACCESS-ESM1-5','CanESM5','CMCC-CM2-SR5','CNRM-CM6-1','FGOALS-g3','GISS-E2-1-G','HadGEM3-GC31-LL','IPSL-CM6A-LR','MIROC6','MPI-ESM1-2-LR','NorESM2-LM']
    elif experiment == 'hist-sol':
        model = ['ACCESS-ESM1-5','CanESM5','GISS-E2-1-G','HadGEM3-GC31-LL','MIROC6','MPI-ESM1-2-LR','NorESM2-LM']
    elif experiment == 'hist-totalO3':
        model = ['CanESM5','GISS-E2-1-G','HadGEM3-GC31-LL','MIROC6','MPI-ESM1-2-LR','NorESM2-LM']
    elif experiment == 'hist-volc':
        model = ['ACCESS-ESM1-5','CanESM5','CMCC-CM2-SR5','GISS-E2-1-G','HadGEM3-GC31-LL','MIROC6','MPI-ESM1-2-LR','NorESM2-LM']

    return model



# In[7]:


#Cropping CVDP data to the North Atlantic sector - requires some shifting of 0 of the lat lon coordinate system.
def CVDP_EM_crop_NA_sector(filename, pattern):
    import xarray as xr
    import numpy as np
    #function which will crop the historical ensemble mean CVDP output to the NA sector
    ds = xr.open_dataset(filename)
    ds = ds[pattern]

    #finding the longitudes that are greater than 180
    new_lon = np.where(ds.lon > 179, ds.lon -360, ds.lon)

    #creating a copy of the data array where the longitudes have been shifted
    ds_shifted = ds.copy()
    ds_shifted.coords['lon'] = new_lon

    #Now need to make sure they are in the correct order and then re-index to make sure the lon get put to match the sorted lon
    sorted_lon = np.sort(ds_shifted.lon)
    ds_shifted = ds_shifted.sel(lon=sorted_lon)

    historical_NAO_EM_shifted = ds_shifted.sel(lat=slice(20,80), lon=slice(-90,40))

    return historical_NAO_EM_shifted


# In[8]:


#Crops to the North Atlantic sector - for the raw LESFMIP data NOT processed by CVDP
def open_cropNA_unitshPA(filename):
    #function to crop an ensemble member to the north atlantic region and put in units of hPa
    import xarray as xr
    data = xr.open_dataset(filename)
    data_NA = data.sel(lat=slice(20,80), lon=slice(-90,40))/100

    return data_NA


# In[9]:


def calculate_spatial_ensemble_mean(file_paths, output_file, variable):
    import xarray as xr
    from dask.diagnostics import ProgressBar
    #Will be passing through an experiment's model's ensembles.
    #opens all the files given by filepath (basically opens all the ensembles)
    ds = xr.open_mfdataset(file_paths, combine='nested', concat_dim='ensemble', parallel=True, chunks={'ensemble':1})[variable]

    # Calculate the mean along the ensemble dimension
    mean = ds.mean(dim='ensemble')    
    
    # Compute the result with a progress bar
    print("Computing the ensemble mean...")
    with ProgressBar():
        mean.compute().to_netcdf(output_file)
    
    print(f'Ensemble mean saved to {output_file}')

    ds.close()
    return mean


# In[6]:


def calculate_seasonal_spatial_ensemble_mean_djf(file_path, var, seas, output_file, year_init, year_final):
    import xarray as xr
    import cftime
    from datetime import datetime
    #opening dataset
    print('in function')
    ds = xr.open_dataset(file_path)

    #checking it is a datetime object
    ds['time'] = xr.decode_cf(ds).time

    calendar = type(ds.time.values[0])

    start,end = get_time_bounds(calendar, year_init, year_final)

    #selecting the psl variable within time bounds
    variable = ds[var].sel(time=slice(start, end))

    #Filter for the desired season (e.g., DJF)
    season_mask = variable.time.dt.season == seas
    ds_months_seas = variable.sel(time=season_mask)

    #assign and adjust year (DJF split over two years so increasing the year of december and then grouping and finding the mean)
    ds_months_seas = ds_months_seas.assign_coords(year=ds_months_seas['time'].dt.year)
    ds_months_seas['year'] = ds_months_seas['year'].where(ds_months_seas['time'].dt.month != 12, ds_months_seas['year'] + 1)
    #ds_months_seas = ds_months_seas.set_coords('year')

    # average over DJF months for each year
    ds_season = ds_months_seas.groupby('year').mean(dim='time')
    ds_season.to_netcdf(output_file)
    print('saved file')
    return ds_season


# In[10]:


#functions defined for calculating the linear trend
def calculate_linear_trend_spat_pattern(file_path, variable, output_file):
    import xarray as xr
    import numpy as np
    from scipy.stats import linregress
    from scipy.stats import t
    # Open dataset and extract variable
    ds = xr.open_dataset(file_path)
    da = ds[variable]

    time = ds['year'].values
    lat = ds['lat'].values
    lon = ds['lon'].values
    time_numeric = np.arange(len(time))

    slope = np.full((len(lat), len(lon)), np.nan)
    intercept = np.full((len(lat), len(lon)), np.nan)
    p_value = np.full((len(lat), len(lon)), np.nan)
    stderr = np.full((len(lat), len(lon)), np.nan)

    for i in range(len(lat)):
        for j in range(len(lon)):
            ts = da[:, i, j].values
            if np.all(np.isfinite(ts)):
                reg = linregress(time_numeric, ts)
                slope[i, j] = reg.slope
                intercept[i, j] = reg.intercept
                p_value[i, j] = reg.pvalue
                stderr[i, j] = reg.stderr

    from scipy.stats import t
    n = len(time_numeric)
    df = n - 2
    alpha = 0.05
    t_crit = t.ppf(1 - alpha/2, df)

    ci_lower = slope - t_crit * stderr
    ci_upper = slope + t_crit * stderr

    slope_da = xr.DataArray(slope, coords=[lat, lon], dims=["lat", "lon"], name="slope")
    intercept_da = xr.DataArray(intercept, coords=[lat, lon], dims=["lat", "lon"], name="intercept")
    p_value_da = xr.DataArray(p_value, coords=[lat, lon], dims=["lat", "lon"], name="p_value")
    ci_lower_da = xr.DataArray(ci_lower, coords=[lat, lon], dims=["lat", "lon"], name="slope_CI_lower")
    ci_upper_da = xr.DataArray(ci_upper, coords=[lat, lon], dims=["lat", "lon"], name="slope_CI_upper")

    # Save to one combined netCDF file
    combined_ds = xr.Dataset({
        "slope": slope_da,
        "intercept": intercept_da,
        "p_value": p_value_da,
        "slope_CI_lower": ci_lower_da,
        "slope_CI_upper": ci_upper_da
    })
    combined_ds.to_netcdf(output_file)


# In[11]:

import numpy as np

def check_eof_orthogonality(EOFs, wgts, tol=1e-10):
    """
    Check whether the two leading EOFs are orthogonal with respect to the given weights.
    
    Parameters
    ----------
    EOFs : xr.DataArray
        EOF patterns with dimensions (mode, lat, lon), at least 2 modes.
    wgts : array-like
        Weights with shape (lat, 1) or (lat,), e.g. sqrt(cos(lat)).
    tol : float
        Numerical tolerance for declaring orthogonality.
    
    Returns
    -------
    orthogonal : bool
        True if the EOFs are orthogonal within tolerance.
    inner : float
        Weighted inner product of EOF1 and EOF2.
    cos_sim : float
        Cosine similarity between EOF1 and EOF2.
    """
    import numpy as np
    EOF1 = EOFs.sel(mode=0)
    EOF2 = EOFs.sel(mode=1)
    
    inner = (EOF1 * EOF2).sum(dim=("lat", "lon"))
    norm1 = np.sqrt((EOF1**2).sum(dim=("lat", "lon")))
    norm2 = np.sqrt((EOF2**2).sum(dim=("lat", "lon")))
    cos_sim = inner / (norm1 * norm2)
    
    return bool(np.isclose(inner, 0, atol=tol)), float(inner), float(cos_sim)


def calculate_regression_map(anomalies, mode, e, m, period, era5=False, individual=False):
    import numpy as np
    import xarray as xr
    from eofs.xarray import Eof
    from eofs.examples import example_data_path

    #function that will calculculate the regression map and EOF for either individual ensemble members or a stack of ensemble members
    #select mode number
    mode_number = 0 if mode == 'NAO' else 1

    #preprocess anomalies
    anomaly_list = [open_cropNA_unitshPA(f) for f in anomalies]

    print('calculating indiv reg maps')

    if individual:
        #calculating the regression map for each ensemble member
        #creating regression map list of filepaths for individual ensembles
        ens_files_reg = []
        ens_files_EOF = []

        for i, da in enumerate(anomaly_list):
            psl = da['psl']

            #creating the weights
            wgts = np.sqrt(np.cos(np.deg2rad(psl.coords['lat'].values)))[..., np.newaxis]
            psl_stacked = psl.rename({'year': 'time'}).transpose('time', 'lat', 'lon')

            #creating the EOF solver
            solver = Eof(psl_stacked, weights=wgts)

            #extracting the EOF pattern
            EOF_pattern = solver.eofs(neofs=2).sel(mode=[0,1])

            #Finding the PC and then normalising (could just keep pcscaling=1 as PC is normalised)
            #pc = solver.pcs(npcs=mode_number+1, pcscaling=1).sel(mode=mode_number)
            #pc = (pc - pc.mean()) / pc.std()

            #calculating the regression map by regressing the psl anomalies onto the pc
            #regression_map = (psl_stacked * pc).mean(dim='time')

            # Sign convention correction via spot check (will change for different models)
            #ensures Icelandic Low is negative and EA main centre is also negative.
            EOF_pattern_NAO = EOF_pattern.sel(mode=0)
            EOF_pattern_EA = EOF_pattern.sel(mode=1)

            output_NAO = anomalies[i].replace('/psl_anomalies', '/regression_patterns/NAO')
            output_EA = anomalies[i].replace('/psl_anomalies', '/regression_patterns/EA')

            if EOF_pattern_NAO.sel(lat=65, lon=-30, method='nearest') > 0 and \
                EOF_pattern_NAO.sel(lat=40, lon=-30, method='nearest') < 0:
                EOF_pattern_NAO *= -1
            output_EOF_NAO = output_NAO.replace('anomaly', 'NAO_EOF_'+period)
            EOF_pattern_NAO.name = 'EOF_NAO_djf'
            EOF_pattern_NAO.to_netcdf(output_EOF_NAO)
                    

            if EOF_pattern_EA.sel(lat=55, lon=-20, method='nearest') > 0 and \
                EOF_pattern_EA.sel(lat=25, lon=-20, method='nearest') < 0:
                EOF_pattern_EA *= -1
            output_EOF_EA = output_EA.replace('anomaly', 'EA_EOF_'+period)
            EOF_pattern_EA.name = 'EOF_EA_djf'
            EOF_pattern_EA.to_netcdf(output_EOF_EA)
                    

            if m == 'era5':
                #creating the filenames for EOF and regression maps for both era5 data (slightly different filepath and naming)
                output_regression_map = output.replace('anomalies_1940-2024', mode+'_regression_map_'+period)
                output_EOF = output.replace('anomalies_1940-2024', mode+'_EOF_'+period)

                print(mode)
                print(output_regression_map)

                #regression_map.name = 'regression_'+mode+'_djf'
                #regression_map.to_netcdf(output_regression_map)

                EOF_pattern.name = 'EOF_'+mode+'_djf'
                EOF_pattern.to_netcdf(output_EOF)

                print('Era5 done')

                return 'era5 done'

        #now calculating the ensemble mean - need to update ens_files to the individual regression maps.
#        print('calculating the mean regression map')

        #opening each ensemble file and storing in one multi-dim array with new dimension ensemble as well as prev lat, lons.
        #ds_reg = xr.open_mfdataset(ens_files_reg, concat_dim='ensemble', combine='nested')
#        ds_EOF = xr.open_mfdataset(ens_files_EOF, concat_dim='ensemble', combine='nested')

        #extracting the nao regression data
        #regression_pattern = ds_reg[f'regression_{mode}_djf']
#        EOF_pattern_NAO = ds_EOF[0]['EOF_NAO_djf']
#        EOF_pattern_EA = ds_EOF[1]['EOF_EA_djf']

        #extracting the max and min from the ensemble spread
        #mean_regression = regression_pattern.mean(dim='ensemble')
#        mean_EOF_NAO = EOF_pattern_NAO.mean(dim='ensemble')
#        mean_EOF_EA = EOF_pattern_EA.mean(dim='ensemble')

#        output_regression_map = '/gws/nopw/j04/extant/users/slbennie/regression_patterns/'+mode+'/'+e+'/'+m+'/psl_mon_'+e+'_'+m+'_DJF_'+mode+'_regression_map_'+period+'.nc'
#        output_EOF_NAO = '/gws/nopw/j04/extant/users/slbennie/regression_patterns/NAO/'+e+'/'+m+'/psl_mon_'+e+'_'+m+'_DJF_NAO_EOF_pattern_'+period+'.nc'
#        output_EOF_EA = '/gws/nopw/j04/extant/users/slbennie/regression_patterns/EA/'+e+'/'+m+'/psl_mon_'+e+'_'+m+'_DJF_EA_EOF_pattern_'+period+'.nc'
        
        #mean_regression.to_netcdf(output_regression_map)
#        mean_EOF_NAO.to_netcdf(output_EOF_NAO)
#        mean_EOF_EA.to_netcdf(output_EOF_EA)


    else:
        all_anomalies = xr.concat(anomaly_list, dim='ensemble')['psl']  # (ensemble, year, lat, lon)
        all_anomalies_stacked = all_anomalies.stack(time=('ensemble', 'year')).reset_index('time', drop=True)
        all_anomalies_stacked = all_anomalies_stacked.transpose('time', 'lat', 'lon')

        coslat = np.cos(np.deg2rad(all_anomalies_stacked.coords['lat'].values)).clip(0., 1.)
        wgts = np.sqrt(coslat)[..., np.newaxis]

        print("Shape of all_anomalies_stacked:", all_anomalies_stacked.shape)

        solver = Eof(all_anomalies_stacked, weights=wgts)
        EOF_pattern = solver.eofs(neofs=2).sel(mode=([0,1]))

        # 1. Get the first two PCs (time series of each mode)
        pcs = solver.pcs(npcs=2, pcscaling=1)  # shape (time, mode)

        # 2. Compute their correlation / covariance
        pc_corr = np.corrcoef(pcs[:, 0], pcs[:, 1])[0, 1]
        print("Correlation between PC1 and PC2:", pc_corr)

        # 3. (Optional) Gram matrix of PCs
        G = np.dot(pcs.T, pcs)  # shape (2,2)
        print("Gram matrix of PCs:\n", G)


        #Check orthogonality
        is_ortho, inner, cos_sim = check_eof_orthogonality(EOF_pattern, wgts)
        print(f"Inner product: {inner:.3e}, Cosine similarity: {cos_sim:.3e}")
        if not is_ortho:
            print("⚠️ Warning: EOF1 and EOF2 are not orthogonal within tolerance!")

        
        output_EOF = '/gws/nopw/j04/extant/users/slbennie/regression_patterns/concatenating/psl_mon_'+e+'_'+m+'_DJF_EOF_pattern_concat_'+period+'3.nc'
        EOF_pattern.to_netcdf(output_EOF)

    return EOF_pattern
        #pc = solver.pcs(npcs=mode_number+1, pcscaling=0).sel(mode=mode_number)
        #pc = (pc - pc.mean()) / pc.std()
        #regression_map = (all_anomalies_stacked * pc).mean(dim='time')

        # Sign convention correction
        #ensures Icelandic Low is negative and EA main centre is also negative.
#        if mode == 'NAO':

#            if regression_map.sel(lat=65, lon=-30, method='nearest') > 0 and \
#               regression_map.sel(lat=40, lon=-30, method='nearest') < 0:
#                regression_map *= -1
#                pc *= -1

#        elif mode == 'EA':

#            if regression_map.sel(lat=55, lon=-20, method='nearest') > 0 and \
#               regression_map.sel(lat=20, lon=-20, method='nearest') < 0:
#                regression_map *= -1
#                pc *= -1        

        #setting up output files paths for the projection and the residual
#        EOF_pattern_NAO = EOF_pattern[0]
#        EOF_pattern_EA = EOF_pattern[1]
        
#        output_EOF_NAO = '/gws/nopw/j04/extant/users/slbennie/regression_patterns/concatenating/psl_mon_'+e+'_'+m+'_DJF_NAO_EOF_pattern_concat_'+period+'.nc'
#        output_EOF_EA = '/gws/nopw/j04/extant/users/slbennie/regression_patterns/concatenating/psl_mon_'+e+'_'+m+'_DJF_EA_EOF_pattern_concat_'+period+'.nc'

#        EOF_pattern_NAO.name = 'EOF_NAO_djf'
#        EOF_pattern_NAO.to_netcdf(output_EOF_NAO)

#        EOF_pattern_EA.name = 'EOF_EA_djf'
#        EOF_pattern_EA.to_netcdf(output_EOF_EA)

        
#        regression_map.name = 'regression_'+mode+'_djf'
#        regression_map.to_netcdf(output_regression_map)
#        EOF_pattern.name = 'EOF_'+mode+'_djf'
#        EOF_pattern.to_netcdf(output_EOF)



# In[9]:


#functions defined for calculating the projections
def project_onto_regression(trend_raw, regression_map, trend_var, mode, e, m, period):
    import xarray as xr
    import numpy as np

    #function which will project a trend (lat,lon) in hPa onto a spatial pattern (lat,lon) hPa to get a single NAO index value
    #will then calculate the residual (trend - mode_congruent part) and saves both the NAO congruent part and the residual
    #in the same folder, output_file is for the NAO/EA_congruent part filename
    #can then change the input spat pattern to calculate the projection onto other eofs, e.g. the EAP


    if isinstance(trend_raw, xr.DataArray):
        trend = trend_raw
    else:
        print('here')
        trend = trend_raw[trend_var]

    print('doing some calcs')
    # Weight psl data by coslat to account for grid cell area decreasing with latitude
    weights = np.cos(np.radians(trend["lat"].values))
    weights_2d = weights[:, np.newaxis]

    # weight psl (or another variable) anomalies by area of each gridcell
    weighted_trend = trend * weights_2d
    weighted_regression = regression_map * weights_2d

    # flatten both of the fields so that they are both 1D
    trend_flat = weighted_trend.stack(spatial=('lat','lon'))
    regression_flat = weighted_regression.stack(spatial=('lat','lon'))

    #replace any NaNs with zeros to stop any weird stuff happening
    trend_flat = trend_flat.fillna(0)
    regression_flat = regression_flat.fillna(0)

    #Now do the dot product which is the projection
    dot_product = (trend_flat * regression_flat).sum().item()

    #calculating the index - or I guess the PC?????
    index = dot_product / (regression_flat**2).sum().item()

    #Now multiplying the pattern by the index and returning that too
    projection = (index * regression_map)
    residual = trend - projection

    projection.name = 'projection_'+mode+'_djf'
    residual.name = 'residual_'+mode+'_djf'

    output_projection = '/gws/nopw/j04/extant/users/slbennie/projection_indicies/NAtlantic_forced_trends/'+e+'/'+m+'/psl_mon_'+e+'_'+m+'_DJF_'+mode+'_projection_'+period+'_EOF.nc'
    output_residual = '/gws/nopw/j04/extant/users/slbennie/projection_indicies/NAtlantic_forced_trends/'+e+'/'+m+'/psl_mon_'+e+'_'+m+'_DJF_'+mode+'_residual_'+period+'_EOF.nc'

    #outputting .nc files for plotting
    projection.to_netcdf(output_projection)
    residual.to_netcdf(output_residual)

    return projection, residual


# In[ ]:




