import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

######one true one false - change pixels, else keep missing#######
def fill_pixels(dem, wo_one_day, inundation_mask, max_iterations):
    filled_wo = np.copy(wo_one_day)  # Copy to avoid modifying the original data
    mask = inundation_mask > 0  # Areas to be filled

    def fill_condition(i, j, dem, filled_wo, mask):
        if not mask[i, j] or not np.isnan(filled_wo[i, j]):
            return False, False  # No need to fill if out of mask or not missing

        window_i_start = max(i - 1, 0)
        window_i_end = min(i + 2, dem.shape[0])
        window_j_start = max(j - 1, 0)
        window_j_end = min(j + 2, dem.shape[1])

        dem_window = dem[window_i_start:window_i_end, window_j_start:window_j_end]
        wo_window = filled_wo[window_i_start:window_i_end, window_j_start:window_j_end]

        # Condition for wet filling: if there's a wet pixel with higher DEM in the window
        wet_condition = np.any((wo_window > 0) & (dem_window >= dem[i, j]))

        # Additional condition for dry filling: if there's a dry pixel with lower DEM in the window
        dry_condition = np.any((wo_window == 0) & (dem_window <= dem[i, j]))

        return wet_condition, dry_condition

    for iteration in range(max_iterations):
        filled_any = False
        for i in reversed(range(dem.shape[0])):
            for j in range(dem.shape[1]):
                wet_condition, dry_condition = fill_condition(i, j, dem, filled_wo, mask)
                
                # Check that exactly one condition is true
                if wet_condition != dry_condition:  # This ensures only one is True and the other is False
                    if wet_condition:
                        filled_wo[i, j] = 1  # Mark as wet
                        filled_any = True
                    else:  # Since wet_condition != dry_condition, this else implies dry_condition is True
                        filled_wo[i, j] = 0  # Mark as dry
                        filled_any = True
        print(f"No. of Iterations: {iteration}")
        if not filled_any:  # Break the loop if no updates
            break

    return filled_wo

# Load the NetCDF data
tiff_file = 'DEM.tif'
ds = xr.open_rasterio(tiff_file)  # Use open_rasterio for GeoTIFF files
ds = ds.rio.write_crs("epsg:4326", inplace=True)

wo_file = 'wo_23yr_masked_4.nc'
mask_file = 'zoomed_inundation_2x_mask.nc'
inundation_mask = xr.open_dataset(mask_file)['flood_depth']

# Add Inundation Information
DEM = ds.sel(band=1).squeeze()

wo_data = xr.open_dataset(wo_file)
date = '2022_11_11'
wo_one_day = wo_data[date]


dem_aligned = DEM.interp(x=inundation_mask.x, y=inundation_mask.y, method='nearest')
wo_one_day_aligned = wo_one_day.interp(x=inundation_mask.x, y=inundation_mask.y, method='nearest')

# Convert DataArrays to numpy arrays for processing
dem_aligned_np = dem_aligned.values
wo_one_day_aligned_np = wo_one_day_aligned.values

# Fill missing pixels in wo_one_day_aligned
filled_wo_one_day_np = fill_pixels(dem_aligned_np, wo_one_day_aligned_np, inundation_mask, 200)

# Convert the filled numpy array back to an xarray DataArray
filled_wo_one_day = xr.DataArray(filled_wo_one_day_np, coords=wo_one_day_aligned.coords, dims=wo_one_day_aligned.dims)

# # Optional: Plot the result
# plt.figure(figsize=(10, 6))
# filled_wo_one_day.plot()
# plt.title('Filled Water Observation for One Day')
# plt.show()

raw_file = 'processed_wo_bourke.nc'
raw_data = xr.open_dataset(raw_file)
wo_original = raw_data[date]
wo_original_aligned = wo_original.interp(x=inundation_mask.x, y=inundation_mask.y, method='nearest')


# Calculate TP, TN, FP, FN only for the updated pixels
TP = ((filled_wo_one_day == 1) & (wo_original_aligned == 1) & np.isnan(wo_one_day_aligned)).sum().item()
TN = ((filled_wo_one_day == 0) & (wo_original_aligned == 0) & np.isnan(wo_one_day_aligned)).sum().item()
FP = ((filled_wo_one_day == 1) & (wo_original_aligned == 0) & np.isnan(wo_one_day_aligned)).sum().item()
FN = ((filled_wo_one_day == 0) & (wo_original_aligned == 1) & np.isnan(wo_one_day_aligned)).sum().item()

# Calculate Critical Success Index (CSI)
CSI = TP / (TP + FN + FP) if (TP + FN + FP) != 0 else np.nan

# Calculate Overall Accuracy
OA = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else np.nan

# Print metrics
print(f"Critical Success Index: {100*CSI:.2f}%")
print(f"Overall Accuracy: {100*OA:.2f}%")

classification_map = filled_wo_one_day.where(inundation_mask)

# Calculate TP, TN, FP, FN only for the updated pixels
TP_loc = ((filled_wo_one_day == 1) & (wo_original_aligned == 1)) & np.isnan(wo_one_day_aligned)
TN_loc = ((filled_wo_one_day == 0) & (wo_original_aligned == 0)) & np.isnan(wo_one_day_aligned)
FP_loc = ((filled_wo_one_day == 1) & (wo_original_aligned == 0)) & np.isnan(wo_one_day_aligned)
FN_loc = ((filled_wo_one_day == 0) & (wo_original_aligned == 1)) & np.isnan(wo_one_day_aligned)

classification_map = xr.where(TP_loc, 2, classification_map)  # TP
classification_map = xr.where(TN_loc, 3, classification_map)  # TN
classification_map = xr.where(FP_loc, 4, classification_map)  # FP
classification_map = xr.where(FN_loc, 5, classification_map)  # FN


# Custom colormap: 0=dry (grey), 1=wet (blue), 2=TP, 3=TN, 4=FP, 5=FN
colors = ['grey', 'blue', 'green', 'lightgrey', 'yellow', 'red']  # Example colors for each category
cmap = mcolors.ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]  # Boundaries between classes
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Extracting x and y limits directly from the DataArray
x = classification_map.x
y = classification_map.y
extent = [x.min().item(), x.max().item(), y.min().item(), y.max().item()]

# Plotting
plt.figure(figsize=(10, 8))
# Make sure the origin is set to 'lower' to match typical geographical maps orientation
im = plt.imshow(classification_map, extent=extent, origin='lower', cmap=cmap, norm=norm, interpolation='none')
plt.gca().invert_yaxis()  # This inverts the y-axis

# Add a colorbar with labels
cbar = plt.colorbar(im, ticks=[0, 1, 2, 3, 4, 5])
cbar.ax.set_yticklabels(['Dry', 'Wet', 'Detected', 'Correct Absence', 'False Alarm', 'Miss'])

plt.title('Classified Results for DEM-filled Data')
plt.show()

# Get the current figure
fig = plt.gcf()

filename = f"bourke_mask_4_hydroDEM_geofabric.png"

# Save the figure to a file
fig.savefig(filename, dpi=300)

filled_wo_one_day.to_netcdf('hydroDEM_mask4_geofabric.nc')
