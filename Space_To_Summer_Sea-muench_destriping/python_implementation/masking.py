#!/usr/bin/env python3
"""
Script to create a binary land_clouds mask based on cloud albedo and land mask conditions.
Mask = 1 where (cloud_albedo > threshold OR pixel overlaps Massachusetts land from GeoJSON) AND l2_flag conditions are met
Mask = 0 where conditions are not met
"""

import os
import numpy as np
from osgeo import gdal, ogr
from typing import Dict


def read_geotiff_band(filepath, band_num=1):
    """Read a single band from a GeoTIFF file."""
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Could not open file: {filepath}")

    band = dataset.GetRasterBand(band_num)
    data = band.ReadAsArray()
    nodata = band.GetNoDataValue()

    # Get spatial reference info
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    dataset = None
    return data, nodata, geotransform, projection


def create_land_mask_from_geojson(geojson_path, reference_dataset):
    """Create a land mask from GeoJSON by rasterizing the geometries."""
    # Open the reference dataset to get dimensions and geospatial info
    ref_ds = gdal.Open(reference_dataset, gdal.GA_ReadOnly)
    if ref_ds is None:
        raise ValueError(f"Could not open reference dataset: {reference_dataset}")

    cols = ref_ds.RasterXSize
    rows = ref_ds.RasterYSize
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    ref_ds = None

    # Open GeoJSON file
    vector_ds = ogr.Open(geojson_path)
    if vector_ds is None:
        raise ValueError(f"Could not open GeoJSON file: {geojson_path}")

    layer = vector_ds.GetLayer()

    # Create in-memory raster for rasterization
    mem_driver = gdal.GetDriverByName('MEM')
    raster_ds = mem_driver.Create('', cols, rows, 1, gdal.GDT_Byte)
    raster_ds.SetGeoTransform(geotransform)
    raster_ds.SetProjection(projection)

    # Initialize with zeros
    band = raster_ds.GetRasterBand(1)
    band.Fill(0)

    # Rasterize the vector data (burn value of 1 for land areas)
    gdal.RasterizeLayer(raster_ds, [1], layer, burn_values=[1])

    # Read the rasterized data
    land_mask = band.ReadAsArray().astype(bool)

    # Cleanup
    raster_ds = None
    vector_ds = None

    return land_mask


def write_mask_geotiff(output_path, mask_data, geotransform, projection):
    """Write a single-band mask to a GeoTIFF file."""
    rows, cols = mask_data.shape

    # Create output dataset
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte)

    if dataset is None:
        raise ValueError(f"Could not create output file: {output_path}")

    # Set geotransform and projection
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)

    # Convert boolean mask to uint8 (0 or 1)
    if mask_data.dtype == bool:
        mask_data = mask_data.astype(np.uint8)

    # Write band
    dataset.GetRasterBand(1).WriteArray(mask_data)

    # Set metadata
    dataset.SetMetadataItem('AREA_OR_POINT', 'Area')

    # Flush and close
    dataset.FlushCache()
    dataset = None


def l2flag_union(l2_flag_masks):
    """Read all l2_flag_masks and compute their union (logical OR)."""
    l2_flag_mask_arrays = []
    for l2_flag_mask_path in l2_flag_masks:
        mask_array, _, _, _ = read_geotiff_band(l2_flag_mask_path, 1)
        l2_flag_mask_arrays.append(mask_array)

    if len(l2_flag_mask_arrays) == 0:
        raise ValueError("No l2_flag_masks provided")
    elif len(l2_flag_mask_arrays) == 1:
        l2_flag_union = l2_flag_mask_arrays[0]
    else:
        # Union: pixel is masked if any mask is nonzero
        l2_flag_union = np.zeros_like(l2_flag_mask_arrays[0], dtype=l2_flag_mask_arrays[0].dtype)
        for arr in l2_flag_mask_arrays:
            l2_flag_union = np.logical_or(l2_flag_union, arr)
        l2_flag_union = l2_flag_union.astype(l2_flag_mask_arrays[0].dtype)

    return l2_flag_union


def create_binary_mask(cloud_albedo, land_geojson_mask, l2_clouds_mask, out_bounds_mask, albedo_threshold, nodata_value=None):
    """
    Create binary mask based on cloud albedo and land mask conditions.
    Mask = 1 where (cloud_albedo > threshold OR pixel is land) AND l2_flag conditions are met
    Mask = 0 where conditions are not met
    """
    # Create mask for valid albedo data (not NoData)
    if nodata_value is not None:
        valid_albedo = (cloud_albedo != nodata_value)
    else:
        valid_albedo = np.ones_like(cloud_albedo, dtype=bool)

    # Initialize binary mask with zeros
    binary_mask = np.zeros_like(cloud_albedo, dtype=np.uint8)

    # Set mask = 1 where cloud albedo > threshold (and data is valid)
    supplemental_cloud_condition = (cloud_albedo > albedo_threshold) & valid_albedo

    # L2 flag condition (requirement that land+cloud is where l2_flag has value 1)
    # out_bounds_mask_condition = (out_bounds_mask == 0)

    # Set mask = 1 where pixel is land OR cloud condition is met, AND l2_flag condition is met
    # mask_condition = (supplemental_cloud_condition | l2_clouds_mask | land_geojson_mask) & out_bounds_mask_condition
    mask_condition = (supplemental_cloud_condition | l2_clouds_mask | land_geojson_mask) & out_bounds_mask
    binary_mask[mask_condition] = 1

    return binary_mask


def create_land_clouds_mask(config: Dict, date_dir_path: str):
    """Main function to create land_clouds mask."""

    # Define paths based on config
    rotation_dir = os.path.join(date_dir_path, config['paths']['rotation_out_rel_path'])
    nc_rotation_dir = os.path.join(rotation_dir, config['rotation']['nc_out_rel_path'])
    masks_dir = os.path.join(date_dir_path, config['paths']['masks_out_rel_path'])

    destriped_land_cloud_reqs_dir = os.path.join(date_dir_path, config['paths']['stripe_correction_out_rel_path'], 'land_cloud_reqs')

    # Input files
    # rgb_path = os.path.join(nc_rotation_dir, config['palette_map']['rgb']['save_name'])
    # cloud_albedo_path = os.path.join(nc_rotation_dir, config['nc_to_gtiff']['nc_exports']['cloud_albedo']['save_name'])
    cloud_albedo_path = os.path.join(destriped_land_cloud_reqs_dir, config['nc_to_gtiff']['nc_exports']['cloud_albedo']['save_name'])
    geojson_path = config['masking']['land_clouds_mask']['massachusetts_geojson_abs_path']

    # Output path
    output_path = os.path.join(masks_dir, config['masking']['land_clouds_mask']['mask_save_rel_path'])

    # Get albedo threshold from config
    albedo_threshold = config['masking']['land_clouds_mask']['cloud_albedo_threshold']

    # Get L2 flag mask paths
    # l2flag_masks = {os.path.join(masks_dir, f"{mask[0]}.tif"):mask[1] for mask in config['masking']['land_clouds_mask']['masks']}

    # out_bounds_mask_path = os.path.join(masks_dir, f"{config['masking']['land_clouds_mask']['out_bounds_mask'][0]}.tif")

    # Read input data
    cloud_albedo, nodata_albedo, geotransform, projection = read_geotiff_band(cloud_albedo_path)

    # Create land mask from GeoJSON
    land_geojson_mask = create_land_mask_from_geojson(geojson_path, cloud_albedo_path)

    l2_clouds_mask_path = os.path.join(masks_dir, f"{config['masking']['land_clouds_mask']['l2_clouds_mask'][0]}.tif")
    out_bounds_mask_path = os.path.join(masks_dir, f"{config['masking']['land_clouds_mask']['out_bounds_mask'][0]}.tif")

    l2_clouds_mask = (read_geotiff_band(l2_clouds_mask_path, 1)[0] == config['masking']['land_clouds_mask']['l2_clouds_mask'][1])

    out_bounds_mask = (read_geotiff_band(out_bounds_mask_path, 1)[0] == config['masking']['land_clouds_mask']['out_bounds_mask'][1])

    # # Create union of L2 flag masks
    # l2flag_mask = l2flag_union(l2flag_masks)

    # Create binary mask
    binary_mask = create_binary_mask(
        cloud_albedo, land_geojson_mask, l2_clouds_mask, out_bounds_mask,
        albedo_threshold, nodata_albedo
    )

    # Write output
    write_mask_geotiff(output_path, binary_mask, geotransform, projection)

    print(f"Created land_clouds mask: {output_path}")
