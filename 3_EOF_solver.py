import os
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray as rio
import numpy as np
import eofs
import matplotlib.colors as mcolors

# Load the NetCDF data
nc_file = 'combined_flood_depth_1x.nc'
ds = xr.open_dataset(nc_file)
ds = ds.rio.write_crs("epsg:4326", inplace=True)

wo_file = 'wo_23yr_masked_1.nc'
raw_file = 'processed_wo_bourke.nc'
mask_file = 'zoomed_inundation_1x_mask.nc'
inundation_mask = xr.open_dataset(mask_file)
threshold = 0.5
optimal_modes = 200
# Add Inundation Information
ds['inundation'] = xr.where(ds['flood_depth'] > 0, 1, 0)

wo_data = xr.open_dataset(wo_file)
raw_data = xr.open_dataset(raw_file)
date = '2022_12_29'
wo_original = raw_data[date]
wo_one_day = wo_data[date]

original_model_data = ds['inundation']

# Mask out pixels outside inundation extent bounds
masked_model_data = original_model_data.where(inundation_mask['flood_depth'] == 1)

# Reshape masked model data into (time, x*y) dimension
model_stacked = np.reshape(masked_model_data.values,(len(original_model_data['time']), -1))

# Record location where model data is not NANs
model_nonan_idx = ~np.isnan(model_stacked)

model_stacked_nonan = model_stacked[:,model_nonan_idx[0]]

# Perform EOF on 2D modelled maps without NANs
solver = eofs.standard.Eof(model_stacked_nonan,center=True)
new_eofs = solver.eofs(neofs=optimal_modes)
np.save(f'zoomed_bom_1x_{optimal_modes}_modes_eofs.npy', new_eofs)

## Prepare WO data
masked_wo_original = wo_original.where(inundation_mask['flood_depth'] == 1)

wo_original_stacked = np.reshape(masked_wo_original.values, (1, -1))

wo_original_extent = wo_original_stacked[:,model_nonan_idx[0]]

masked_wo_one_day = wo_one_day.where(inundation_mask['flood_depth'] == 1)

# Reshape WO observation into (1, x*y) dimension
wo_one_day_stacked = np.reshape(masked_wo_one_day.values,(1, -1))

# Remove all pixels outside inundation extent
wo_stacked_extent = wo_one_day_stacked[:,model_nonan_idx[0]]

# Record location where is not cloudy
cloudy_idx = ~np.isnan(wo_stacked_extent)

# Mask out cloudy pixels in spatial modes
partial_new_eofs = new_eofs[:,cloudy_idx[0]]

data_mean_stacked = np.mean(model_stacked_nonan, axis=0)
np.save('zoomed_1x_data_mean.npy', data_mean_stacked)