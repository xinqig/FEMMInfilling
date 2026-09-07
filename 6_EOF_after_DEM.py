import os
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray as rio
import numpy as np
import eofs
import matplotlib.colors as mcolors
import geopandas as gpd

polygons = gpd.read_file('Study_Area/Zoomed_study_area.shp')
wo_file = 'hydroDEM_mask2_geofabric.nc'
raw_file = 'filtered_wo_bourke.nc'
mask_file = 'zoomed_inundation_2x_mask.nc'
sample_file = 'wo_23yr_masked_2.nc'
inundation_mask = xr.open_dataset(mask_file)
threshold = 0.5
optimal_modes = 200

wo_data = xr.open_dataset(wo_file)
raw_data = xr.open_dataset(raw_file)
wo_missing_data = xr.open_dataset(sample_file)
date = '2022_11_11'
wo_original = raw_data[date]
wo_missing = wo_missing_data[date]
wo_one_day = wo_data

masked_wo_missing = wo_missing.where(inundation_mask['flood_depth'] == 1)
# Reshape masked model data into (time, x*y) dimension
inundation_mask_stacked = np.reshape(inundation_mask['flood_depth'].values, (1,-1))

# Record location where model data is not NANs
model_nonan_idx = (inundation_mask_stacked == 1)

new_eofs = np.load('zoomed_bom_2x_200_modes_eofs.npy')
## Prepare WO data
masked_wo_original = wo_original.where(inundation_mask['flood_depth'] == 1)

wo_original_stacked = np.reshape(masked_wo_original.values, (1, -1))

wo_original_extent = wo_original_stacked[:,model_nonan_idx[0]]

masked_wo_one_day = wo_one_day['__xarray_dataarray_variable__'].where(inundation_mask['flood_depth'] == 1)

# Reshape WO observation into (1, x*y) dimension
wo_one_day_stacked = np.reshape(masked_wo_one_day.values,(1, -1))

# Remove all pixels outside inundation extent
wo_stacked_extent = wo_one_day_stacked[:,model_nonan_idx[0]]

# Record location where is not cloudy
cloudy_idx = ~np.isnan(wo_stacked_extent)

# Mask out cloudy pixels in spatial modes
partial_new_eofs = new_eofs[:,cloudy_idx[0]]

# data_mean_stacked = np.mean(model_stacked_nonan, axis=0)
data_mean_stacked = np.load('zoomed_2x_data_mean.npy')
# data_mean_reshaped = np.reshape(data_mean.values, (1, -1))
# data_mean_stacked = data_mean_reshaped[:,model_nonan_idx[0]]
centered_wo = wo_stacked_extent - data_mean_stacked

centered_wo_nonan = centered_wo[:,cloudy_idx[0]]

wo_pcs_trans, _, _, _ = np.linalg.lstsq(partial_new_eofs.T, centered_wo_nonan.T, rcond=None)

# Reshape calculated_pc to have shape (1, M)
wo_pcs = wo_pcs_trans.reshape(1, optimal_modes)

reconstructed_data = wo_pcs.dot(new_eofs) + data_mean_stacked

# Initialize variables for the loop
binary_reconstructed = xr.where(reconstructed_data > threshold, 1, 0)

# Update the missing data areas with reconstructed values
updating_data = xr.where(~cloudy_idx[0], binary_reconstructed, wo_stacked_extent)

wo_stacked_empty = np.full(wo_one_day_stacked.shape, np.nan)
wo_stacked_empty[:,model_nonan_idx[0]]= updating_data
wo_stacked_updated = np.where(np.isnan(wo_stacked_empty), wo_one_day_stacked, wo_stacked_empty)

# If wo_one_day is an xarray DataArray or similar, you can directly extract its shape
y, x = masked_wo_one_day.shape

# Reshape wo_stacked_updated back to the original shape of wo_one_day
wo_updated_reshaped = wo_stacked_updated.reshape(y, x)

wo_updated_da = xr.DataArray(wo_updated_reshaped, dims=masked_wo_one_day.dims, coords=masked_wo_one_day.coords)

classification_map = wo_updated_da

# Calculate TP, TN, FP, FN only for the updated pixels
TP_loc = ((wo_updated_da == 1) & (masked_wo_original == 1)) & np.isnan(masked_wo_missing)
TN_loc = ((wo_updated_da == 0) & (masked_wo_original == 0)) & np.isnan(masked_wo_missing)
FP_loc = ((wo_updated_da == 1) & (masked_wo_original == 0)) & np.isnan(masked_wo_missing)
FN_loc = ((wo_updated_da == 0) & (masked_wo_original == 1)) & np.isnan(masked_wo_missing)

classification_map = xr.where(TP_loc, 2, classification_map)  # TP
classification_map = xr.where(TN_loc, 3, classification_map)  # TN
classification_map = xr.where(FP_loc, 4, classification_map)  # FP
classification_map = xr.where(FN_loc, 5, classification_map)  # FN

# colors = ['dimgray', 'navy', 'lightskyblue', 'lightgray', 'crimson', 'gold']

# Custom colormap: 0=dry (grey), 1=wet (blue), 2=TP, 3=TN, 4=FP, 5=FN
colors = ['grey', 'blue', 'green', 'lightgrey', 'yellow', 'red']  # Example colors for each category
cmap = mcolors.ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]  # Boundaries between classes
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Extracting x and y limits directly from the DataArray
x = classification_map.x
y = classification_map.y
extent = [x.min().item(), x.max().item(), y.max().item(), y.min().item()]


# Plotting
plt.figure(figsize=(10, 8))
# Make sure the origin is set to 'lower' to match typical geographical maps orientation
im = plt.imshow(classification_map, extent=extent, origin='lower', cmap=cmap, norm=norm, interpolation='none')
# polygons.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=1)  # Adjust color and linewidth as needed

# im = plt.imshow(classification_map.sel(x=slice(x_min, x_max), y=slice(y_min, y_max)),
#                 extent=extent,
#                 origin='lower', cmap=cmap, norm=norm, interpolation='none')

plt.gca().invert_yaxis()  # This inverts the y-axis

# Add a colorbar with labels
cbar = plt.colorbar(im, ticks=[0, 1, 2, 3, 4, 5])
cbar.ax.set_yticklabels(['Dry', 'Inundated', 'Detected', 'Correct Dry', 'False Alarm', 'Missed'])

plt.title('Classified Results for Reconstructed Data')
plt.show()


# Calculate TP, TN, FP, FN only for the updated pixels
TP = TP_loc.sum().item()
TN = TN_loc.sum().item()
FP = FP_loc.sum().item()
FN = FN_loc.sum().item()

# Calculate Critical Success Index (CSI)
CSI = TP / (TP + FN + FP) if (TP + FN + FP) != 0 else np.nan

# Calculate Overall Accuracy
OA = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else np.nan

# Print metrics
print(f"Critical Success Index: {100*CSI:.2f}%")
print(f"Overall Accuracy: {100*OA:.2f}%")



# mask_no = int(wo_file.split('_')[-1].split('.')[0])

# # # Get the current figure
# fig = plt.gcf()

# filename = f"bourke_mask_{mask_no}_{date}_{optimal_modes}_modes_dem.png"

# # # Save the figure to a file
# fig.savefig(filename, dpi=300)

sample_file = 'wo_23yr_masked_4.nc'
wo_missing_data = xr.open_dataset(sample_file)
date = '2022_11_11'
wo_missing = wo_missing_data[date]
data = wo_missing.where(inundation_mask['flood_depth'] == 1)
fig, ax = plt.subplots(figsize=(12, 8))
cmap = plt.cm.colors.ListedColormap(['grey', 'blue']) # Dry as white, Wet as blue
bounds = [-0.5, 0.5, 1.5]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
cbar_labels = ['Dry', 'Inundated']
plt.rcParams.update({'font.size': 16})
polygons.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=1)  # Adjust color and linewidth as needed

im = data.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)

ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
plt.gca().invert_yaxis()
# polygons.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=1)  # Adjust color and linewidth as needed
plt.title(' ')
plt.xlabel(' ')
plt.ylabel(' ')
plt.show()
