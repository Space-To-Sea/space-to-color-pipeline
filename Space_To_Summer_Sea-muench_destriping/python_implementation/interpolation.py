#!/usr/bin/env python

import os
import glob
import shutil
import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Tuple, Optional
import warnings
import subprocess
import tempfile

def interpolate_gtiff_gdal(input_image_path: str, output_path: str,
                            verbose: bool,
                            band: int = 1, max_distance: int = 100,
                            smoothing_iterations: int = 0):
    """
    Advanced version with more options for gdal_fillnodata.py

    Parameters:
    -----------
    input_image_path : str
        Path to input GeoTIFF file
    nodata_value : float
        The nodata value to fill
    output_path : str
        Path for output filled GeoTIFF
    verbose : bool
        Print verbose information
    mask_path : str, optional
        Path to mask file defining areas to fill
    band : int, default=1
        Band number to process (1-based)
    max_distance : int, default=100
        Maximum distance (in pixels) to search for valid pixels
    smoothing_iterations : int, default=0
        Number of 3x3 smoothing filter passes
    """

    # Get basic info about the input file
    with rasterio.open(input_image_path) as src:
        profile = src.profile
        actual_nodata = src.nodata

    if verbose:
        print("File nodata value:", actual_nodata)
        # print("Specified nodata value:", nodata_value)
        print("Processing band:", band)

    # Prepare gdal_fillnodata.py command
    cmd = [
        "gdal_fillnodata.py",
        "-md", str(max_distance),
        "-si", str(smoothing_iterations),
        "-b", str(band)
    ]

    # Add input and output files
    cmd.extend([input_image_path, output_path])

    if verbose:
        print("Running command:", " ".join(cmd))

    try:
        # Run gdal_fillnodata.py
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if verbose:
            print("Command output:", result.stdout)
            if result.stderr:
                print("Command stderr:", result.stderr)
        print(f"Successfully filled nodata values. Output saved to: {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error running gdal_fillnodata.py: {e}")
        print(f"Command that failed: {' '.join(cmd)}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        raise

def interpolate_file(input_file_path, output_dir, output_file_basename, num_iterations, max_distance, smoothing_iterations, verbose):
    intermediate_basename = "_filled_intermediate"
    # nodata_val = -32767
    input_path = os.path.join(output_dir, f"{output_file_basename}_intermediate_input.tif")

    shutil.copy(input_file_path, input_path)

    for i in range(num_iterations):
        intermediate_path = os.path.join(output_dir, f"{output_file_basename}{intermediate_basename}_{i}.tif")
        interpolate_gtiff_gdal(
            input_image_path=input_path,
            output_path=intermediate_path,
            verbose=verbose,
            max_distance=max_distance,
            smoothing_iterations=smoothing_iterations
        )
        os.remove(input_path)
        input_path = intermediate_path

    output_file_path = os.path.join(output_dir, f"{output_file_basename}.tif")

    shutil.copy(intermediate_path, output_file_path)
    os.remove(input_path)


# def apply_out_bounds_mask(interpolated_file_path: str, out_bounds_mask_path: str, verbose: bool = False):
#     """
#     Apply out_bounds_mask to interpolated file, setting pixels to nodata where
#     out_bounds_mask is 1 or nodata.

#     Parameters:
#     -----------
#     interpolated_file_path : str
#         Path to interpolated GeoTIFF file
#     out_bounds_mask_path : str
#         Path to out_bounds_mask GeoTIFF file
#     verbose : bool
#         Print verbose information
#     """
#     if verbose:
#         print(f"Applying out_bounds_mask to: {interpolated_file_path}")

#     # Read the interpolated file
#     with rasterio.open(interpolated_file_path, 'r+') as img_src:
#         img_data = img_src.read(1)
#         img_nodata = img_src.nodata
#         img_profile = img_src.profile

#         # Read the out_bounds_mask
#         with rasterio.open(out_bounds_mask_path) as mask_src:
#             mask_data = mask_src.read(1)
#             mask_nodata = mask_src.nodata

#             if verbose:
#                 print(f"Image nodata value: {img_nodata}")
#                 print(f"Mask nodata value: {mask_nodata}")
#                 print(f"Image shape: {img_data.shape}")
#                 print(f"Mask shape: {mask_data.shape}")

#             # Check that dimensions match
#             if img_data.shape != mask_data.shape:
#                 raise ValueError(f"Image and mask dimensions don't match: {img_data.shape} vs {mask_data.shape}")

#             # Create mask condition: where out_bounds_mask is 1 or nodata
#             if mask_nodata is not None:
#                 mask_condition = (mask_data == 1) | (mask_data == mask_nodata)
#             else:
#                 mask_condition = (mask_data == 1)

#             # Count pixels that will be masked
#             pixels_to_mask = np.sum(mask_condition)
#             if verbose:
#                 print(f"Pixels to be masked: {pixels_to_mask} out of {img_data.size}")

#             # Apply mask: set pixels to nodata where mask condition is True
#             if img_nodata is not None:
#                 img_data[mask_condition] = img_nodata
#             else:
#                 # If no nodata value defined, use a default
#                 if img_data.dtype.kind == 'f':  # float
#                     img_data[mask_condition] = np.nan
#                 else:  # integer
#                     # Use a value outside the typical range
#                     img_data[mask_condition] = -32767
#                     # Update the profile to set nodata
#                     img_profile.update(nodata=-32767)
#                     img_src.nodata = -32767

#             # Write the modified data back
#             img_src.write(img_data, 1)

#             if verbose:
#                 print(f"Successfully applied out_bounds_mask to {interpolated_file_path}")


def apply_out_bounds_mask(interpolated_file_path: str, out_bounds_mask_path: str, mode: str, verbose: bool = False):
    """
    Apply out_bounds_mask to interpolated file, setting pixels to nodata where
    out_bounds_mask is 1 or nodata.

    Parameters:
    -----------
    interpolated_file_path : str
        Path to interpolated GeoTIFF file
    out_bounds_mask_path : str
        Path to out_bounds_mask GeoTIFF file
    verbose : bool
        Print verbose information
    """
    if verbose:
        print(f"Applying out_bounds_mask to: {interpolated_file_path}")

    # Read the interpolated file
    with rasterio.open(interpolated_file_path, 'r+') as img_src:
        img_data = img_src.read(1)
        img_nodata = img_src.nodata
        img_profile = img_src.profile

        # Read the out_bounds_mask
        with rasterio.open(out_bounds_mask_path) as mask_src:
            mask_data = mask_src.read(1)
            mask_nodata = mask_src.nodata

            if verbose:
                print(f"Image nodata value: {img_nodata}")
                print(f"Mask nodata value: {mask_nodata}")
                print(f"Image shape: {img_data.shape}")
                print(f"Mask shape: {mask_data.shape}")

            # Check that dimensions match
            if img_data.shape != mask_data.shape:
                raise ValueError(f"Image and mask dimensions don't match: {img_data.shape} vs {mask_data.shape}")

            # Create mask condition: where out_bounds_mask is 1 or nodata
            if mask_nodata is not None:
                mask_condition = (mask_data == 1) | (mask_data == mask_nodata)
            else:
                mask_condition = (mask_data == 1)

            if mode == "empty_rows_only":
                # Identify rows where every value is nodata in the image
                if mask_nodata is not None:
                    empty_rows = np.all((mask_data == mask_nodata) | (mask_data == 1), axis=1)
                    # Create a mask: True for all pixels in empty rows
                    mask_condition = mask_condition & empty_rows[:, np.newaxis]
                else:
                    if verbose:
                        print("Image has no nodata value defined.")

            # Count pixels that will be masked
            pixels_to_mask = np.sum(mask_condition)
            if verbose:
                print(f"Pixels to be masked: {pixels_to_mask} out of {img_data.size}")

            # Apply mask: set pixels to nodata where mask condition is True
            if img_nodata is not None:
                img_data[mask_condition] = img_nodata
            else:
                # If no nodata value defined, use a default
                if img_data.dtype.kind == 'f':  # float
                    img_data[mask_condition] = np.nan
                else:  # integer
                    # Use a value outside the typical range
                    img_data[mask_condition] = -32767
                    # Update the profile to set nodata
                    img_profile.update(nodata=-32767)
                    img_src.nodata = -32767

            # Write the modified data back
            img_src.write(img_data, 1)

            if verbose:
                print(f"Successfully applied out_bounds_mask to {interpolated_file_path}")

def interpolation(config: Dict, date_dir_path: str):
    nc_file_rel_paths_to_interpolate = {
        "chlor_a": config['nc_to_gtiff']['nc_exports']['chlor_a']['save_name'],
        "diatoms_hirata": config['nc_to_gtiff']['nc_exports']['diatoms_hirata']['save_name'],
        "greenalgae_hirata": config['nc_to_gtiff']['nc_exports']['greenalgae_hirata']['save_name'],
        "dinoflagellates_hirata": config['nc_to_gtiff']['nc_exports']['dinoflagellates_hirata']['save_name'],
        "prymnesiophytes_hirata": config['nc_to_gtiff']['nc_exports']['prymnesiophytes_hirata']['save_name'],
    }

    nc_export_dir_path = os.path.join(
        date_dir_path,
        config['paths']['rotation_out_rel_path'],
        config['rotation']['nc_out_rel_path']
    )

    num_iterations = config['interpolation']['num_iterations']
    max_distance = config['interpolation']['max_distance']
    smoothing_iterations = config['interpolation']['smoothing_iterations']
    verbose = config['verbose']['interpolation']
    interpolation_export_dir_path = os.path.join(date_dir_path, config['paths']['interpolation_out_rel_path'])

    if os.path.exists(interpolation_export_dir_path):
        shutil.rmtree(interpolation_export_dir_path, ignore_errors=True)
    os.makedirs(interpolation_export_dir_path)

    for var, nc_file_save_rel_path in nc_file_rel_paths_to_interpolate.items():
        input_file_path = os.path.join(nc_export_dir_path, nc_file_save_rel_path)
        interpolate_file(input_file_path, interpolation_export_dir_path, var, num_iterations, max_distance, smoothing_iterations, verbose)

    masking_mode = config['masking']['apply_out_bounds_mask_to_interpolated']

    if masking_mode != "none":
        if verbose:
            print("Applying out_bounds_mask to interpolated files...")

        # Get the out_bounds_mask file path
        masks_dir = os.path.join(date_dir_path, config['paths']['masks_out_rel_path'])
        out_bounds_mask_file = config['masking']['out_bounds_mask_file']
        out_bounds_mask_path = os.path.join(masks_dir, out_bounds_mask_file)

        if os.path.exists(out_bounds_mask_path):
            # Apply mask to each interpolated file
            for var in nc_file_rel_paths_to_interpolate.keys():
                interpolated_file_path = os.path.join(interpolation_export_dir_path, f"{var}.tif")
                if os.path.exists(interpolated_file_path):
                    apply_out_bounds_mask(interpolated_file_path, out_bounds_mask_path, masking_mode, verbose)
                else:
                    if verbose:
                        print(f"Warning: Interpolated file not found: {interpolated_file_path}")
        else:
            print(f"Warning: out_bounds_mask file not found: {out_bounds_mask_path}")
            if verbose:
                print("Skipping out_bounds_mask application")
