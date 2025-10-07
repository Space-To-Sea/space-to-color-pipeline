import os
import shutil

import rasterio
from typing import Dict
import numpy as np
from rasterio.warp import reproject, Resampling
from affine import Affine

def rotate_geotiff(input_path, output_path, angle=14.3, crop=False):
    """
    Rotate a GeoTIFF image while maintaining georeferencing accuracy AND preserving metadata.

    Parameters:
    -----------
    input_path : str
        Path to input GeoTIFF file
    output_path : str
        Path to output rotated GeoTIFF file
    angle : float, default=14.3
        Angle of rotation in degrees
    crop : bool, default=False
        If True, crop to original dimensions. If False, expand to fit entire rotated image.

    Returns:
    --------
    None
        Saves rotated GeoTIFF to output_path
    """

    with rasterio.open(input_path) as src:
        print(f"Rotating {input_path=} and saving at {output_path=}")

        # Get the original transform and CRS
        src_transform = src.transform
        crs = src.crs

        # Read all bands
        bands = []
        print(f"{src.count=}")
        for i in range(1, src.count + 1):
            print(f"Appending band {i}")
            bands.append(src.read(i))

        # Get dimensions
        print(f"Number of bands: {len(bands)}")
        print(f"Shape of first band: {bands[0].shape}")
        height, width = bands[0].shape

        if crop:
            # Keep original dimensions
            dst_width = width
            dst_height = height
            print(f"Cropping to original size: {dst_width} x {dst_height}")
        else:
            # Calculate expanded dimensions to fit entire rotated image
            angle_rad = np.radians(angle)
            cos_a = abs(np.cos(angle_rad))
            sin_a = abs(np.sin(angle_rad))
            dst_width = int(np.ceil(width * cos_a + height * sin_a))
            dst_height = int(np.ceil(width * sin_a + height * cos_a))
            print(f"Expanding to fit rotation: {dst_width} x {dst_height}")

        # Calculate transform for proper centering
        center_x, center_y = width / 2.0, height / 2.0
        new_center_x, new_center_y = dst_width / 2.0, dst_height / 2.0

        translate_to_center = Affine.translation(center_x, center_y)
        rotate_transform = Affine.rotation(angle)
        translate_back = Affine.translation(-new_center_x, -new_center_y)

        pixel_transform = translate_to_center * rotate_transform
        dst_transform = src_transform * pixel_transform * translate_back

        # Set properties for output file (this preserves dtype, nodata, etc.)
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({
            "transform": dst_transform,
            "height": dst_height,
            "width": dst_width,
        })

        print(f"Preserving dtype: {dst_kwargs['dtype']}")
        print(f"Preserving nodata: {dst_kwargs['nodata']}")

        # Write rotated image to disk
        with rasterio.open(output_path, "w", **dst_kwargs) as dst:
            # Copy global metadata tags
            dst.update_tags(**src.tags())

            # Reproject each band
            for band_idx, band_data in enumerate(bands, 1):
                reproject(
                    source=band_data,
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src_transform,
                    src_crs=crs,
                    dst_transform=dst_transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata
                )

                band_tags = src.tags(band_idx)
                dst.update_tags(band_idx, **band_tags)
                print(f"Copied metadata for band {band_idx}: {band_tags}")

# def rotate_geotiff(input_path, output_path, angle=14.3, crop=False):
#     """
#     Rotate a GeoTIFF image while maintaining georeferencing accuracy AND preserving metadata.

#     Parameters:
#     -----------
#     input_path : str
#         Path to input GeoTIFF file
#     output_path : str
#         Path to output rotated GeoTIFF file
#     angle : float, default=14.3
#         Angle of rotation in degrees

#     Returns:
#     --------
#     None
#         Saves rotated GeoTIFF to output_path
#     """

#     with rasterio.open(input_path) as src:
#         print(f"Rotating {input_path=} and saving at {output_path=}")

#         # Get the original transform and CRS
#         src_transform = src.transform
#         crs = src.crs

#         # Create affine transformations for rotation and translation
#         rotate = Affine.rotation(angle)

#         # Combine affine transformations
#         dst_transform = src_transform * rotate

#         # Read all bands
#         bands = []
#         print(f"{src.count=}")
#         for i in range(1, src.count + 1):
#             print(f"Appending band {i}")
#             bands.append(src.read(i))

#         # Get dimensions and adjust if needed
#         print(f"Number of bands: {len(bands)}")
#         print(f"Shape of first band: {bands[0].shape}")
#         height, width = bands[0].shape

#         angle_rad = np.radians(angle)
#         cos_a = abs(np.cos(angle_rad))
#         sin_a = abs(np.sin(angle_rad))

#         dst_width = int(np.ceil(width * cos_a + height * sin_a))
#         dst_height = int(np.ceil(width * sin_a + height * cos_a))

#         # And update the transform to center properly:
#         center_x, center_y = width / 2.0, height / 2.0
#         new_center_x, new_center_y = dst_width / 2.0, dst_height / 2.0

#         translate_to_center = Affine.translation(center_x, center_y)
#         rotate_transform = Affine.rotation(angle)
#         translate_back = Affine.translation(-new_center_x, -new_center_y)

#         pixel_transform = translate_to_center * rotate_transform

#         dst_transform = src_transform * pixel_transform * translate_back

#         # Set properties for output file
#         dst_kwargs = src.meta.copy()
#         dst_kwargs.update({
#             "transform": dst_transform,
#             "height": dst_height,
#             "width": dst_width,
#         })

#         # Write rotated image to disk
#         with rasterio.open(output_path, "w", **dst_kwargs) as dst:
#             # Copy global metadata tags
#             dst.update_tags(**src.tags())

#             # Reproject each band
#             for band_idx, band_data in enumerate(bands, 1):
#                 reproject(
#                     source=band_data,
#                     destination=rasterio.band(dst, band_idx),
#                     src_transform=src_transform,
#                     src_crs=crs,
#                     dst_transform=dst_transform,
#                     dst_crs=crs,
#                     resampling=Resampling.bilinear
#                 )

#                 # CRITICAL: Copy band-level metadata tags after writing each band
#                 band_tags = src.tags(band_idx)
#                 dst.update_tags(band_idx, **band_tags)
#                 print(f"Copied metadata for band {band_idx}: {band_tags}")

# def rotate_geotiff(input_path, output_path, angle=14.3):
#     """
#     Rotate a GeoTIFF image while maintaining georeferencing accuracy.

#     Parameters:
#     -----------
#     input_path : str
#         Path to input GeoTIFF file
#     output_path : str
#         Path to output rotated GeoTIFF file
#     angle : float, default=14.3
#         Angle of rotation in degrees
#     shift_x : int, default=0
#         Shift in x direction to adjust positioning
#     shift_y : int, default=0
#         Shift in y direction to adjust positioning
#     adj_width : int, default=0
#         Adjust width of output raster if needed
#     adj_height : int, default=0
#         Adjust height of output raster if needed

#     Returns:
#     --------
#     None
#         Saves rotated GeoTIFF to output_path
#     """

#     with rasterio.open(input_path) as src:
#         print(f"Rotating {input_path=} and saving at {output_path=}")

#         # Get the original transform and CRS
#         src_transform = src.transform
#         crs = src.crs

#         # Create affine transformations for rotation and translation
#         rotate = Affine.rotation(angle)

#         # Combine affine transformations
#         dst_transform = src_transform * rotate

#         # Read all bands
#         bands = []
#         print(f"{src.count=}")
#         for i in range(1, src.count + 1):
#             print(f"Appending band {i}")
#             bands.append(src.read(i))

#         # Get dimensions and adjust if needed
#         print(f"Number of bands: {len(bands)}")
#         print(f"Shape of first band: {bands[0].shape}")
#         height, width = bands[0].shape

#         angle_rad = np.radians(angle)
#         cos_a = abs(np.cos(angle_rad))
#         sin_a = abs(np.sin(angle_rad))

#         dst_width = int(np.ceil(width * cos_a + height * sin_a))
#         dst_height = int(np.ceil(width * sin_a + height * cos_a))

#         # And update the transform to center properly:
#         center_x, center_y = width / 2.0, height / 2.0
#         new_center_x, new_center_y = dst_width / 2.0, dst_height / 2.0

#         #translate_to_center = Affine.translation(-center_x, -center_y)
#         translate_to_center = Affine.translation(center_x, center_y)
#         rotate_transform = Affine.rotation(angle)
#         #translate_back = Affine.translation(0, -new_center_y)
#         translate_back = Affine.translation(-new_center_x, -new_center_y)

#         pixel_transform = translate_to_center * rotate_transform #* translate_back

#         dst_transform = src_transform * pixel_transform * translate_back

#         # Set properties for output file
#         dst_kwargs = src.meta.copy()
#         dst_kwargs.update({
#             "transform": dst_transform,
#             "height": dst_height,
#             "width": dst_width,
#         })

#         # Write rotated image to disk
#         with rasterio.open(output_path, "w", **dst_kwargs) as dst:
#             # Reproject each band
#             for band_idx, band_data in enumerate(bands, 1):
#                 reproject(
#                     source=band_data,
#                     destination=rasterio.band(dst, band_idx),
#                     src_transform=src_transform,
#                     src_crs=crs,
#                     dst_transform=dst_transform,
#                     dst_crs=crs,
#                     resampling=Resampling.bilinear
#                 )

def clip_geotiff_values(input_path, output_path, flag=False):

    # Clip the input file
    with rasterio.open(input_path) as src:
        data = src.read(1)
        metadata = src.tags(1)
        print(f"{os.path.basename(input_path)}: {metadata=}")
        if 'valid_min' not in metadata or 'valid_max' not in metadata:
            valid_min = 0.
            valid_max = 1.
        else:
            valid_min = float(metadata['valid_min'])
            valid_max = float(metadata['valid_max'])
        nodata_value = src.nodata
        print(f"{os.path.basename(input_path)}: {valid_min} to {valid_max}, nodata: {nodata_value}")

        ############################################################################################

        if flag:
            unique_before = np.unique(data)
            print(f"Unique values BEFORE clipping: {unique_before}")
            print(f"Data type BEFORE clipping: {data.dtype}")

        ############################################################################################

        # if nodata_value is not None:
        #     valid_mask = data != nodata_value
        # else:
        #     valid_mask = np.ones_like(data, dtype=bool)

        # data[valid_mask] = np.clip(data[valid_mask], min_val, max_val)

        data = np.where(data < valid_min, nodata_value, data)  # Values < valid_min → nodata
        data = np.where(data > valid_max, valid_max, data)     # Values > valid_max → valid_max

        # data = np.clip(data, valid_min, valid_max)

        print(f"Clipped values to range: {valid_min} to {valid_max}")

        ############################################################################################

        if flag:
            unique_after = np.unique(data)
            print(f"Unique values AFTER clipping: {unique_after}")
            print(f"Data type AFTER clipping: {data.dtype}")

        print(f"Clipped values to range: {valid_min} to {valid_max}")

        ############################################################################################

        # Write clipped data
        with rasterio.open(output_path, 'w', **src.meta) as dst:
            dst.write(data, 1)
            # Copy band-level metadata tags
            dst.update_tags(1, **metadata)

def rotate_gtiffs(config: Dict, date_dir_path: str):

    verbose = config['verbose']['rotation']

    output_dir_path = os.path.join(date_dir_path, config['paths']['rotation_out_rel_path'])

    nc_input_dir_path = os.path.join(date_dir_path, config['paths']['nc_to_gtiff_out_rel_path'])
    nc_output_dir_path = os.path.join(output_dir_path, config['rotation']['nc_out_rel_path'])
    mask_input_dir_path = os.path.join(date_dir_path, config['paths']['flags_out_rel_path'], config['flags']['masks_out_rel_path'])
    mask_output_dir_path = os.path.join(output_dir_path, config['rotation']['mask_out_rel_path'])

    date_dir_masks_dir_path = os.path.join(date_dir_path, config['paths']['masks_out_rel_path'])

    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path, ignore_errors=True)

    os.makedirs(output_dir_path)

    os.makedirs(nc_output_dir_path)
    os.makedirs(mask_output_dir_path)

    if os.path.exists(date_dir_masks_dir_path):
        shutil.rmtree(date_dir_masks_dir_path, ignore_errors=True)

    os.makedirs(date_dir_masks_dir_path)

    # Rotate and clip every .tif file in input_dir_path
    for file_name in os.listdir(nc_input_dir_path):
        if file_name.lower().endswith('.tif') and file_name != config['nc_to_gtiff']['nc_exports']['l2_flags']['save_name']:
            input_file_path = os.path.join(nc_input_dir_path, file_name)
            clipped_file_path = os.path.join(nc_output_dir_path, f"clipped_{file_name}")
            rotated_file_path = os.path.join(nc_output_dir_path, f"rotated_{file_name}")
            final_file_path = os.path.join(nc_output_dir_path, file_name)

            # clip_geotiff_values(input_file_path, clipped_file_path)

            # Rotate the GeoTIFF
            # rotate_geotiff(clipped_file_path, rotated_file_path, angle=config['rotation']['angle'])
            rotate_geotiff(input_file_path, rotated_file_path, angle=config['rotation']['angle'])

            # Clip the rotated GeoTIFF
            clip_geotiff_values(rotated_file_path, final_file_path)

            os.remove(rotated_file_path)

    for file_name in os.listdir(mask_input_dir_path):
        print(file_name)
        if file_name.lower().endswith('.tif'):
            input_file_path = os.path.join(mask_input_dir_path, file_name)
            rotated_file_path = os.path.join(mask_output_dir_path, f"rotated_{file_name}")
            final_file_path = os.path.join(mask_output_dir_path, file_name)

            # Rotate the GeoTIFF
            # rotate_geotiff(input_file_path, rotated_file_path, angle=config['rotation']['angle'])
            rotate_geotiff(input_file_path, final_file_path, angle=config['rotation']['angle'])

            # clip_geotiff_values(rotated_file_path, final_file_path, True)

            #os.symlink(final_file_path, os.path.join(date_dir_masks_dir_path, file_name))
            shutil.copy2(final_file_path, os.path.join(date_dir_masks_dir_path, file_name))

    return None
