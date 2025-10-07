import os
import shutil
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import pyproj
from datetime import datetime
from typing import Dict
import json

# def create_bbox_geojson(bounds, region_num):
#     """Create a GeoJSON FeatureCollection for a bounding box."""

#     corners_x = [bounds['west'], bounds['east'], bounds['east'], bounds['west']]
#     corners_y = [bounds['south'], bounds['south'], bounds['north'], bounds['north']]

#     corners = zip(corners_x, corners_y)

#     coordinates = [corners]

#     return [{
#             "type": "Feature",
#             "properties": {"region": region_num},
#             "geometry": {
#                 "type": "Polygon",
#                 "coordinates": coordinates
#             }
#         }]

def create_bbox_geojson(bounds, region_num):
    """Create a GeoJSON FeatureCollection for a bounding box."""

    # Create coordinates for a closed polygon (clockwise)
    corners_x = [bounds['west'], bounds['east'], bounds['east'], bounds['west'], bounds['west']]
    corners_y = [bounds['south'], bounds['south'], bounds['north'], bounds['north'], bounds['south']]

    # Convert zip object to list for JSON serialization
    corners = list(zip(corners_x, corners_y))

    # GeoJSON Polygon coordinates format: [[[x,y],[x,y],...]]
    coordinates = [corners]

    return [{
        "type": "Feature",
        "properties": {"region": region_num},
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates
        }
    }]

def save_debug_geojson(bounds, csv_output_dir, region_num, crs, file_name):
    """Save debug GeoJSON files for bounding boxes."""

    features = []

    bbox_feature = create_bbox_geojson(bounds, region_num)
    features.extend(bbox_feature)

    output_path = os.path.join(csv_output_dir, f"{file_name}.geojson")

    geojson_data = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)

def transform_bounds_to_dict(bounds, transformer):
    """Transform bounding box coordinates and return as a bounds dictionary."""
    corners_x = [bounds['west'], bounds['east'], bounds['east'], bounds['west']]
    corners_y = [bounds['south'], bounds['south'], bounds['north'], bounds['north']]

    transformed_x, transformed_y = transformer.transform(corners_x, corners_y)

    return {
        'west': min(transformed_x),
        'east': max(transformed_x),
        'south': min(transformed_y),
        'north': max(transformed_y)
    }

def load_tiff_data(input_dir_path, variables, bounds):
    """Load TIFF data for all required variables within specified bounds."""
    tiff_data = {}
    transform = None
    crs = None

    for variable in variables:
        tiff_path = os.path.join(input_dir_path, f"{variable}.tif")

        if not os.path.exists(tiff_path):
            continue

        with rasterio.open(tiff_path) as src:
            window = from_bounds(
                bounds['west'], bounds['south'],
                bounds['east'], bounds['north'],
                src.transform
            )

            data = src.read(1, window=window)
            window_transform = src.window_transform(window)

            if transform is None:
                transform = window_transform
                crs = src.crs

            tiff_data[variable] = data

    return tiff_data, transform, crs

def create_coordinate_arrays(data_shape, window_transform, transformer):
    """Create coordinate arrays for the data."""
    height, width = data_shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))

    # Convert pixel coordinates to source CRS coordinates
    x_coords, y_coords = rasterio.transform.xy(window_transform, rows.flatten(), cols.flatten())
    x_coords = np.array(x_coords).reshape(height, width)
    y_coords = np.array(y_coords).reshape(height, width)

    # Transform to target CRS (geographic coordinates)
    lon_flat, lat_flat = transformer.transform(x_coords.flatten(), y_coords.flatten())
    lon = np.array(lon_flat).reshape(height, width)
    lat = np.array(lat_flat).reshape(height, width)

    return lon, lat

def create_csv_data(tiff_data, variables, lon, lat):
    """Create CSV data from TIFF data and coordinates."""
    if not tiff_data:
        return []

    first_var = list(tiff_data.keys())[0]
    height, width = tiff_data[first_var].shape
    rows = []
    feature_id = 0

    for i in range(height):
        for j in range(width):
            lon_val = lon[i, j]
            lat_val = lat[i, j]

            if np.isnan(lon_val) or np.isnan(lat_val) or np.isinf(lon_val) or np.isinf(lat_val):
                continue

            row_data = {}
            has_valid_data = True

            for variable in variables:
                if variable not in tiff_data:
                    has_valid_data = False
                    break

                value = tiff_data[variable][i, j]

                if np.isnan(value) or np.isinf(value):
                    has_valid_data = False
                    break

                row_data[variable] = float(value)

            if has_valid_data:
                row = {
                    'featureId': feature_id,
                    **row_data,
                    'longitude': float(lon_val),
                    'latitude': float(lat_val)
                }
                rows.append(row)
                feature_id += 1

    return rows

def write_csv_file(rows, variables, csv_path, scene_width, pixel_resolution_km):
    """Write CSV data to file with proper SeaDAS format."""
    if not rows:
        return

    df = pd.DataFrame(rows)
    column_order = ['featureId'] + variables + ['longitude', 'latitude']
    df = df[column_order]

    header_row = "featureId " + " ".join([f"{var}:float" for var in variables]) + " longitude:float latitude:float"

    with open(csv_path, 'w') as f:
        f.write(f"#sceneRasterWidth={scene_width}\n")
        f.write(f"#rasterResolutionInKm={pixel_resolution_km}\n")
        f.write(f"{header_row}\n")
        df.to_csv(f, sep='\t', index=False, header=False, float_format='%.8f')

def calculate_scene_metadata(tiff_data, window_transform):
    """Calculate scene metadata for CSV headers."""
    first_var = list(tiff_data.keys())[0]
    height, width = tiff_data[first_var].shape

    pixel_size_meters = abs(window_transform[0])
    pixel_resolution_km = pixel_size_meters / 1000.0

    return width, pixel_resolution_km

def csv_export(config: Dict, date_dir_path: str):
    """Export CSV files for specified regions from GeoTIFF files."""

    verbose = config['verbose']['csv_export']

    # Configuration
    csv_config = config['csv_export']
    regions = csv_config['regions']
    region_crs = csv_config['region_crs']
    netcdf_crs = csv_config['netcdf_crs']

    # Get variables from nc_to_gtiff exports (excluding l2_flags)
    variables = [
        key for key, val in config['nc_to_gtiff']['nc_exports'].items()
        if key != 'l2_flags'
    ]

    # Paths
    input_dir_path = os.path.join(date_dir_path, config['paths']['nc_to_gtiff_out_rel_path'])
    output_dir_path = os.path.join(date_dir_path, config['paths']['csv_out_rel_path'])

    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path, ignore_errors=True)

    os.makedirs(output_dir_path)

    # Extract and format date
    folder_name = os.path.basename(date_dir_path)
    date_obj = datetime.strptime(folder_name, "%m_%d_%Y")
    date_str = date_obj.strftime("%Y-%m-%d")

    # Create coordinate transformers
    transformer = pyproj.Transformer.from_crs(region_crs, netcdf_crs, always_xy=True)
    inverse_transformer = pyproj.Transformer.from_crs(netcdf_crs, "EPSG:4326", always_xy=True)

    # Process each region
    for region_num, original_bounds in regions.items():

        # Transform bounds from region CRS to netcdf CRS
        transformed_bounds = transform_bounds_to_dict(original_bounds, transformer)

        if verbose:
            save_debug_geojson(original_bounds, output_dir_path, region_num, region_crs, f"original_crs_bbox_{region_num}")
            save_debug_geojson(transformed_bounds, output_dir_path, region_num, netcdf_crs, f"transformed_crs_bboxes_{region_num}")

        # Load TIFF data using the transformed bounds
        tiff_data, window_transform, crs = load_tiff_data(
            input_dir_path, variables, transformed_bounds
        )

        if not tiff_data:
            continue

        # Create coordinate arrays - convert from netcdf CRS back to geographic
        first_var = list(tiff_data.keys())[0]
        lon, lat = create_coordinate_arrays(
            tiff_data[first_var].shape, window_transform, inverse_transformer
        )

        # Create CSV data
        rows = create_csv_data(tiff_data, variables, lon, lat)

        if not rows:
            continue

        # Calculate scene metadata
        scene_width, pixel_resolution_km = calculate_scene_metadata(tiff_data, window_transform)

        # Write CSV file
        csv_filename = f"{date_str}-r{region_num}.csv"
        csv_path = os.path.join(output_dir_path, csv_filename)
        write_csv_file(rows, variables, csv_path, scene_width, pixel_resolution_km)

    return output_dir_path
