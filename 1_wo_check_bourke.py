import os
import glob
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import mapping
import re
import rioxarray as rio
import numpy as np

def wo_process(file_name):
    projection = "EPSG:4326"
    
    # Open the GeoTIFF file into an xarray Dataset
    dataset = xr.open_rasterio(file_name)
    original_nodata = dataset.rio.nodata
    dataset = dataset.rio.write_nodata(None)

    if dataset.rio.crs != projection:
        dataset = dataset.rio.reproject(projection)
        
    dataset = dataset.rio.write_nodata(original_nodata)

    # Generate masks
    wet_mask = (dataset[0].values >> 7) % 2 == 1
    other_bits_mask = (dataset[0].values & 0x7F) > 0  # Mask where any of the first 7 bits are set
    missing_mask = other_bits_mask

    # Determine wet, dry and missing
    result_array = np.where(wet_mask, 1, np.where(missing_mask, np.nan, 0))
    result_data = xr.DataArray(result_array, coords=dataset[0].coords, dims=dataset[0].dims)

    return result_data


def process_tiffs(base_dir, wo_process):
    # Find all .tif files in the base directory
    tiff_files = glob.glob(os.path.join(base_dir, "*.tif"))
    wo_data = []

    # Process each .tiff file
    for tiff_file in tiff_files:
        # Extract date from the filename
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        date_match = re.search(date_pattern, tiff_file)
        if date_match:
            extracted_date = date_match.group(0)
        else:
            extracted_date = "Unknown Date"
        
        # assuming that `wo_process` is a defined function that processes your tif files
        single_wo_data = wo_process(tiff_file)
        single_wo_data.attrs['date'] = extracted_date  # add the date as an attribute
        wo_data.append(single_wo_data)
    
    return wo_data


# import pandas as pd

# # List to store the counts and percentages for each wo_data item
# data_list = []

# for data in wo_data:
#     missing_count = np.isnan(data.values).sum()
#     dry_count = (data.values == 0).sum()
#     wet_count = (data.values == 1).sum()
#     total_pixels = data.values.size
    
#     missing_percentage = (missing_count / total_pixels) * 100
#     dry_percentage = (dry_count / total_pixels) * 100
#     wet_percentage = (wet_count / total_pixels) * 100

#     data_list.append([data.attrs['date'], missing_count, dry_count, wet_count, 
#                       missing_percentage, dry_percentage, wet_percentage])

# # Convert data list to a pandas DataFrame
# df = pd.DataFrame(data_list, columns=['Date', 'Missing Count', 'Dry Count', 'Wet Count',
#                                       'Missing Percentage', 'Dry Percentage', 'Wet Percentage'])

# # Calculate the summary values
# total_images = len(wo_data)
# wet_images = (df['Wet Percentage'] > 1).sum()

# # Append summary row to the dataframe
# df.loc['Summary'] = ['-', '-', '-', 'No. Total Images', total_images, 'No. Wet Images', wet_images]
# # df.to_csv('WO_bourke_stats.csv', index=False)

# # Display the dataframe
# print(df)

def wo_plot(data, place):
    date = data.attrs['date']
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = plt.cm.colors.ListedColormap(['grey', 'blue']) # Dry as white, Wet as blue
    bounds = [-0.5, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    cbar_labels = ['Dry', 'Inundated']
    plt.rcParams.update({'font.size': 16})
    im = data.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], orientation='vertical', fraction=0.03)
    cbar.ax.set_yticklabels(cbar_labels)
    plt.title(f'Water Observations in {place} - {date}')
    plt.show()



def filter_data(wo_data, non_missing_threshold):

    wo_filtered_data = []  # List to store filtered data

    for data in wo_data:
        missing_count = np.isnan(data.values).sum()
        total_pixels = data.values.size
        non_missing_percentage = 100 - ((missing_count / total_pixels) * 100)

        # Add the data to the list only if the non-missing pixels meet or exceed the threshold
        if non_missing_percentage >= non_missing_threshold:
            wo_filtered_data.append(data)

    return wo_filtered_data


