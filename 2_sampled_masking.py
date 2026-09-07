import os
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray as rio
import numpy as np
import eofs
import matplotlib.colors as mcolors
import wo_check_bourke as wo
import data_management as dm


import geopandas as gpd

polygons = gpd.read_file('Study_Area/Zoomed_study_area.shp')

wo_data = wo.process_tiffs('Sample_masks', wo.wo_process)

boundary_name = os.path.join("clipped_boundary", "clipped_boundary.shp")
wo_clipped = []
for i in range(len(wo_data)):
    wo_clipped.append(dm.clip_data(wo_data[i], boundary_name))

mask_file = 'zoomed_inundation_2x_mask.nc'

inundation_mask = xr.open_dataset(mask_file)


# masked_wo_missing = wo_missing.where(inundation_mask['flood_depth'] == 1)

# raw_file = 'filtered_wo_bourke.nc'
# raw_data = xr.open_dataset(raw_file)

# mask = [raw_data['2016_10_01'], raw_data['2022_01_11'], 
#         raw_data['2016_10_09'], raw_data['2022_11_27']]


x_min, y_min, x_max, y_max = inundation_mask.rio.bounds()

# Clip and align the wo_data to the extent and grid of ds
aligned_wo_data = []
for data in wo_clipped:
    # First, clip to the extent of ds
    clipped_data = data.rio.clip_box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max)
    
    # Now, align the grids
    aligned_data = clipped_data.rio.reproject_match(inundation_mask)
    
    aligned_wo_data.append(aligned_data)


#####################
mask = 3
#####################

new_missing = xr.where(np.isnan(aligned_wo_data[mask]),True, False)

date = aligned_wo_data[mask].attrs['date']
fig, ax = plt.subplots(figsize=(12, 9))
cmap = plt.cm.colors.ListedColormap(['gray', 'white']) # Dry as white, Wet as blue
bounds = [-0.5, 0.5, 1.5]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
# cbar_labels = ['Observation', 'Sample Mask']
plt.rcParams.update({'font.size': 16})
im = new_missing.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
# cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], orientation='vertical', fraction=0.03)
# cbar.ax.set_yticklabels(cbar_labels)
binary_mask = inundation_mask['flood_depth'] == 1
from scipy.ndimage import binary_closing

# Use morphological closing to fill small gaps within the binary mask
# Adjust the structure size (3x3 here) as needed for the size of gaps you're dealing with
closed_mask = binary_closing(binary_mask, structure=np.ones((1,1)))

# Assuming your DataArray has 'lon' and 'lat' coordinates, prepare them for contour plotting
lon, lat = np.meshgrid(inundation_mask['x'], inundation_mask['y'])

CS = ax.contour(lon, lat, closed_mask, levels=[0.5], colors='blue', linewidths=3)
import matplotlib.patches as mpatches
# inundation_areas = inundation_mask['flood_depth'] == 1
# inundation_mask['flood_depth'].where(inundation_areas).plot(ax=ax, cmap='Blues', alpha=0.5, add_colorbar=False)
plt.title(f'Sampled Mask {4} for Bourke Extracted at {date}', fontsize=24)
# plt.xlabel('Longitude', fontsize=20)
# plt.ylabel('Latitude', fontsize=20)
# Create colored patches
ax.set_xlabel('')
ax.set_ylabel('')
# ax.set_xticks([])
# ax.set_yticks([])
sample_mask_patch = mpatches.Patch(color='white', label='Sample Mask')
observation_patch = mpatches.Patch(color='grey', label='Observation')
study_area_patch = mpatches.Patch(edgecolor='blue', facecolor='none', linewidth=1, label='Study Area')
# Add these patches to a legend on your plot
plt.legend(handles=[sample_mask_patch, observation_patch, study_area_patch], loc='lower right',facecolor='lightgrey')
plt.show()
