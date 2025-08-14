#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import cftime
import pandas as pd
from datetime import datetime
import matplotlib.colors as mcolors
from scipy.stats import linregress
from eofs.xarray import Eof
from eofs.examples import example_data_path

#My Functions
import importlib
import trend_projection_functions2
importlib.reload(trend_projection_functions2)
from trend_projection_functions2 import get_time_bounds
from trend_projection_functions2 import get_models_for_experiment
from trend_projection_functions2 import CVDP_EM_crop_NA_sector#
from trend_projection_functions2 import open_cropNA_unitshPA
from trend_projection_functions2 import calculate_spatial_ensemble_mean
from trend_projection_functions2 import calculate_seasonal_spatial_ensemble_mean_djf
from trend_projection_functions2 import calculate_linear_trend_spat_pattern
from trend_projection_functions2 import calculate_regression_map
from trend_projection_functions2 import project_onto_regression

#!jupyter nbconvert --to script trend_projection_functions2.ipynb

#notes find a way to update period so its nicer.
#also could fix folder so that I use like a home dir etc.
#anomalies have already been calculated separetly for each model so preusming they are all correct this should run?...
home = '/gws/nopw/j04/extant/users/slbennie/'
variable = 'psl'
period = '1850-2015'
experiments = ['hist-sol','hist-totalO3','hist-volc']
model = ['GISS-E2-1-G']
modes = ['NAO', 'EA']
seas = 'DJF'


for e in experiments:
    #model = get_models_for_experiment(e)
    print(model)
    for m in model:
        if e == 'historical':
            #REGRESSION MAPS
            print('Calculating the regression map for model:', m)

            #path to the folder containing historical experiment and model's psl anomalies (calculated seperatly)
            folder_path = f'{home}psl_anomalies/historical/{m}/'

            #creating the list of files for this experiment and model's psl anomalies
            ens_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if m in filename and '_EM' not in filename]
            #print(ens_files)

            calculate_regression_map(ens_files, modes[0], 'historical', m, period, individual=True)
            calculate_regression_map(ens_files, modes[1], 'historical', m, period, individual=True)

        #opening historical regression maps (mean) for use later - one per model. Use historical for all experiments.
        regression_NAO = xr.open_dataset(f'{home}regression_patterns/{modes[0]}/historical/{m}/{variable}_mon_historical_{m}_{seas}_{modes[0]}_regression_map_{period}.nc')
        regression_EA = xr.open_dataset(f'{home}regression_patterns/{modes[1]}/historical/{m}/{variable}_mon_historical_{m}_{seas}_{modes[1]}_regression_map_{period}.nc')

        #ENSEMBLE MEANS
        print('Calculating the spatial ensemble means and trends for model:', m)
        print('Experiment:', e)
        #getting the LESFMIP file paths
        folder_path = f'/gws/nopw/j04/leader_epesc/CMIP6_SinglForcHistSimul/InterpolatedFlds/psl/{e}/{m}/'
        file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if e in filename and m in filename]

        #creating output files for the mean across ensembles and the djf mean
        ens_mean_file = f'{home}ens_mean_spat/psl/{e}/{m}/{variable}_mon_{e}_{m}_spatial_EM.nc'
        ens_mean_djf_file = ens_mean_file.replace('spatial_EM', 'spatial_'+seas+'_EM_'+period)

        #find spatial mean across the ensembles
        print('Calculating the ensemble mean')
        calculate_spatial_ensemble_mean(file_paths, ens_mean_file, variable)

        #To use once spatial ensemble mean is calculated
        print(f'Calculating the seasonal {seas} spatial ensemble mean')
        calculate_seasonal_spatial_ensemble_mean_djf(ens_mean_file, variable, seas, ens_mean_djf_file, 1850, 2015)


        #LINEAR TREND
        #find the file for the ensemble spatial mean djf
        folder_path = f'{home}ens_mean_spat/psl/{e}/{m}/'
        print(folder_path)

        #selecting file with the correct seas and period
        file_path = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if seas in filename and period in filename][0]
        print(file_path)
        output_file = f'{home}trend_calc_LESFMIP/linear_regression/NAO/{e}/{m}/{variable}_mon_{e}_{m}_{seas}_linear_trend_{period}_stats.nc'

        calculate_linear_trend_spat_pattern(file_path, variable, output_file)


        #PROJECTION
        print("Calculating the projection of the forced trend onto historical regression map calculated from all ensembles:", e)

        #now setting up the folder path to get the file names for each experiment's model's forced response's trend
        folder_path = f'{home}trend_calc_LESFMIP/linear_regression/NAO/{e}/{m}/'

        #Getting the list of file names within the models folder, should only be one trend per model (working off the ensemble means for each model)
        ens_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if m in filename and period in filename]
        #print(ens_files[0], len(ens_files), output_file)

        for i in range(0,len(ens_files)):
            #cropping each trend to just the NA sector and whichever time
            trend = open_cropNA_unitshPA(ens_files[i])#, 1850,2014)

            #multiplying the trend by 165 to convert to units of hPa (currently in units of hPa/year, trend calculated between 1850-2015)
            trend = trend * 165

            #calling the projection functions
            proj_NAO, residual_NAO = project_onto_regression(trend, regression_NAO['regression_NAO_djf'], 'slope', 'NAO', e, m, period)
            proj_EA, residual_EA = project_onto_regression(residual_NAO, regression_EA['regression_EA_djf'], 'residual_NAO_djf', 'EA', e, m, period)





