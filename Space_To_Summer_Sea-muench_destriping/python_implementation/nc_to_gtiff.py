import os
import subprocess
import glob
import shutil

import uuid
import rasterio
from typing import Dict
import numpy as np

def extract_watermask_bounds(watermask_tif_path: str, verbose: bool):
    """
    Extract ul_x, ul_y, lr_x, lr_y coordinates from a GeoTIFF file.

    Args:
        geotiff_file_path (str): Path to the GeoTIFF file

    Returns:
        tuple: (ul_x, ul_y, lr_x, lr_y) coordinates for georeferencing
    """
    try:
        with rasterio.open(watermask_tif_path) as src:
            bounds = src.bounds

            # Extract coordinates
            ul_x = bounds.left
            ul_y = bounds.bottom
            lr_x = bounds.right
            lr_y = bounds.top

            if verbose:
                print(f"DEBUG: Extracted bounds from water mask: {watermask_tif_path}")
                print(f"DEBUG:  Upper Left (ul_x, ul_y): ({ul_x:.8f}, {ul_y:.8f})")
                print(f"DEBUG:  Lower Right (lr_x, lr_y): ({lr_x:.8f}, {lr_y:.8f})")
                print(f"DEBUG:  CRS: {src.crs}")
                print(f"DEBUG:  Transform: {src.transform}")
            return ul_x, ul_y, lr_x, lr_y
    except FileNotFoundError:
        raise FileNotFoundError(f"water mask file not found: {watermask_tif_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading water mask file: {e}")

def translate_warp_netcdf(config: Dict, nc_path: str, intermediate_dir_path: str, output_file_path: str, netcdf_variable: str, upper_left_x, upper_left_y, lower_right_x, lower_right_y):
    """
    Translate and warp a NetCDF file to GeoTIFF format.

    Args:
        input_file_path (str): Path to input NetCDF file
        output_file_path (str): Path to output GeoTIFF file
        netcdf_variable (str): NetCDF variable to extract
        input_crs (str): Input coordinate reference system
        upper_left_x (float): Upper left X coordinate
        upper_left_y (float): Upper left Y coordinate
        lower_right_x (float): Lower right X coordinate
        lower_right_y (float): Lower right Y coordinate
        intermediate_files_dir (str): Directory for temporary files

    Returns:
        int: 0 if successful, 1 if error
    """

    input_file_path = nc_path
    input_crs = config['nc_to_gtiff']['gis']['input_crs']
    intermediate_files_dir_path = intermediate_dir_path
    pixel_size = config['nc_to_gtiff']['gis']['pixel_size']
    output_crs = config['nc_to_gtiff']['gis']['output_crs']

    # Set up temporary file path
    temp_file_path = os.path.join(intermediate_files_dir_path, f"temp_{netcdf_variable.replace('/', '_')}_{uuid.uuid4().hex[:8]}.tif")

    # Remove old temporary file if it exists
    if os.path.exists(temp_file_path):
        print("Removing old temporary_file.tif")
        os.remove(temp_file_path)

    try:
        # First command: gdal_translate
        translate_cmd = [
            "gdal_translate",
            "-a_srs", input_crs,
            "-oo", "HONOUR_VALID_RANGE=NO",
            "-strict",
            "-stats",
            "-a_ullr", str(upper_left_x), str(upper_left_y),
                      str(lower_right_x), str(lower_right_y),
            f"NETCDF:{input_file_path}:{netcdf_variable}",
            temp_file_path
        ]

        subprocess.run(translate_cmd, check=True)

        # Second command: gdalwarp
        warp_cmd = [
            "gdalwarp",
            "-t_srs", output_crs,
            "-tr", pixel_size, pixel_size,
            temp_file_path,
            output_file_path
        ]

        subprocess.run(warp_cmd, check=True)

        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return 0

    except subprocess.CalledProcessError as e:
        print(f"Error running GDAL command: {e}")
        # Clean up temporary file on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Clean up temporary file on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return 1

def nc_to_gtiff(config: Dict, date_dir_path: str):

    verbose = config['verbose']['nc_to_gtiff']

    watermask_tif_path = glob.glob(os.path.join(date_dir_path, config['paths']['watermask_tif_path_search']))[0]
    nc_path = os.path.join(date_dir_path, config['paths']['seadas_l2gen_netcdf_rel_path'])
    output_dir_path = os.path.join(date_dir_path, config['paths']['nc_to_gtiff_out_rel_path'])
    intermediate_dir_path = output_dir_path

    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path, ignore_errors=True)

    os.makedirs(output_dir_path)

    ul_x, ul_y, lr_x, lr_y = extract_watermask_bounds(watermask_tif_path, verbose)

    for _, variable_config in config['nc_to_gtiff']['nc_exports'].items():

        var_name = variable_config['var_name']
        output_file_path = os.path.join(output_dir_path, variable_config['save_name'])

        #################################################################################

        output_file_path = os.path.join(output_dir_path, variable_config['save_name'])

        #################################################################################

        translate_warp_netcdf(config, nc_path, intermediate_dir_path, output_file_path, var_name, ul_x, ul_y, lr_x, lr_y)
        if variable_config.get('valid_min', {}):
            print(f"Sending all negative values to 0 for {var_name=}")
            with rasterio.open(output_file_path, 'r+') as dst:
                data = dst.read(1)
                mask = data < variable_config['valid_min']
                data[mask] = -32767
                dst.write(data, 1)

    return output_dir_path
