import rasterio
import os
import numpy as np
from pathlib import Path 
import rioxarray as rxr
import hydra


def get_band_min_max(folder_path):
    # Get list of raster files in folder
    raster_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')]

    # Define band groups (based on the monthly pattern)
    num_months = 5
    bands_per_month = 6
    total_bands = num_months * bands_per_month
    
    band_mins = {i: [] for i in range(1, total_bands + 1)}
    band_maxes = {i: [] for i in range(1, total_bands + 1)}

    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            for band in range(1, total_bands + 1):
                data = src.read(band).astype(np.float32)        
                band_mins[band].append(np.min(data))
                band_maxes[band].append(np.max(data))

    return band_mins, band_maxes


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    ROOT_PATH = Path(cfg.paths.ROOT_PATH)
    train_imgs_path = ROOT_PATH/'tra_scene'
    train_imgs = list((train_imgs_path).glob("*.tif"))

    mins, maxes = get_band_min_max(train_imgs_path)

    new_min = {i: [] for i in range(1,31)}
    new_max = {i: [] for i in range(1,31)}

    for key, band in mins.items():
        new_min[key] = np.min(band) 

    for key, band in maxes.items():
        new_max[key] = np.max(band)

    # Groups of keys
    groups = [[i + j * 6 for j in range(5)] for i in range(1, 7)]

    # Compute minimum for each group
    min_values = {tuple(group): min(new_min[key] for key in group) for group in groups}

    # Compute max for each group
    max_values = {tuple(group): min(new_max[key] for key in group) for group in groups}

    min_arr = np.array(list(min_values.values()))
    max_arr = np.array(list(max_values.values()))

    return min_arr, max_arr