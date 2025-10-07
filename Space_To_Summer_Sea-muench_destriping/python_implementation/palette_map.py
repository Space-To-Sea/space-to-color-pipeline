import rasterio
import os
import numpy as np
from typing import Dict, List
from rasterio.enums import ColorInterp
import shutil

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_vectorized_color_ramp(palette_values: List[float], palette_colors: List[str], verbose: bool):
    """Create vectorized color interpolation function."""
    palette_values = np.array(palette_values, dtype=np.float32)
    rgb_colors = np.array([hex_to_rgb(color) for color in palette_colors], dtype=np.float32)

    def vectorized_interpolate(data):
        """Vectorized interpolation for entire data arrays."""
        data = data.astype(np.float32)
        result = np.zeros((*data.shape, 3), dtype=np.uint8)
        valid_mask = ~np.isnan(data)

        if not np.any(valid_mask):
            return result

        valid_data = data[valid_mask]

        for channel in range(3):
            interpolated_channel = np.interp(
                valid_data, palette_values, rgb_colors[:, channel]
            )
            result[valid_mask, channel] = interpolated_channel.astype(np.uint8)

        return result

    return vectorized_interpolate

def make_alpha_band_from_mask(date_dir_path: str, config: Dict, masks_dict: Dict, alpha_mask_name: str, verbose: bool) -> np.ndarray:
    """Create alpha band from mask definition."""

    mask = masks_dict[alpha_mask_name][0]

    # Convert to alpha values (255 = opaque, 0 = transparent)
    alpha_255 = mask * 255

    return alpha_255

def create_mask_from_definition(date_dir_path: str, config: Dict, mask_definition: List, verbose: bool) -> np.ndarray:
    """Create boolean mask from mask definition list."""
    mask_conditions = []

    for mask_name, expected_value in mask_definition:
        mask_path = os.path.join(date_dir_path, config['paths']['masks_out_rel_path'], f"{mask_name}.tif")
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
            condition = (mask_data == expected_value)
            mask_conditions.append(condition)

    # Combine all conditions with AND logic
    if len(mask_conditions) == 1:
        final_mask = mask_conditions[0]
    else:
        final_mask = np.logical_and.reduce(mask_conditions)

    return final_mask.astype(np.uint8)

def write_rgba_image(output_path: str, rgb_array: np.ndarray, alpha_band: np.ndarray,
                    profile: Dict, verbose: bool) -> None:
    """Write RGBA image to file."""
    rgba_array = np.zeros((4, rgb_array.shape[1], rgb_array.shape[2]), dtype=np.uint8)
    rgba_array[0:3] = rgb_array
    rgba_array[3] = alpha_band

    profile.update({
        'count': 4,
        'dtype': 'uint8',
        'nodata': None,
    })

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(rgba_array)
        dst.colorinterp = [
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha
        ]

        # Set band statistics
        for band in range(1, 5):
            dst.update_tags(band, **{})
            band_data = rgba_array[band-1]
            dst.update_tags(band,
                           STATISTICS_MINIMUM=int(np.min(band_data)),
                           STATISTICS_MAXIMUM=int(np.max(band_data)))

def save_mask(date_dir_path: str, config: Dict, mask_data: np.ndarray, save_path: str, reference_profile: Dict):
    """Save mask to file."""
    profile = reference_profile.copy()
    profile.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': None,
    })

    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(mask_data, 1)

def process_masks(date_dir_path: str, config: Dict, verbose: bool):
    """Process and save masks as defined in config."""
    masks_dir = os.path.join(date_dir_path, config['paths']['masks_out_rel_path'])
    palette_map_dir = os.path.join(date_dir_path, config['paths']['palette_map_out_rel_path'])

    mask_dict = {}

    # Get reference profile from any existing mask file
    reference_mask_path = os.path.join(date_dir_path,config['paths']['masks_out_rel_path'], f"is_out_bounds.tif")
    with rasterio.open(reference_mask_path) as ref:
        reference_profile = ref.profile

    for mask_name, mask_config in config['palette_map']['masks'].items():
        mask_data = create_mask_from_definition(date_dir_path, config, mask_config['masks'], verbose)

        # Save in masks directory if specified
        if mask_config.get('mask_save_rel_path'):
            save_path = os.path.join(masks_dir, f"{mask_config['mask_save_rel_path']}.tif")
            save_mask(date_dir_path, config, mask_data, save_path, reference_profile)

        # Save in palette_map directory if specified
        if mask_config.get('save_rel_path'):
            save_path = os.path.join(palette_map_dir, f"{mask_config['save_rel_path']}.tif")
            save_mask(date_dir_path, config, mask_data, save_path, reference_profile)

        mask_dict[mask_name] = [mask_data, reference_profile]

    return mask_dict

def apply_palette_to_data(data: np.ndarray, palette_config: Dict, palette_name: str,
                         chlor_a_config: Dict, verbose: bool) -> np.ndarray:
    """Apply color palette to data array."""
    palette_info = palette_config['palette']
    palette_values = [float(val) for val in palette_info['values']]
    palette_colors = palette_info['colors']

    if palette_name == 'grayscale':
        valid_mask = (data > -30000.0) & (~np.isnan(data))
        valid_data = data[valid_mask]
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)

        data_normalized = (data - data_min) / (data_max - data_min)
        data_normalized = np.clip(data_normalized, 0, 1)
        palette_min, palette_max = float(palette_values[0]), float(palette_values[-1])
        data_for_palette = data_normalized * (palette_max - palette_min) + palette_min

    elif palette_name == 'oceancolor':
        range_min = chlor_a_config.get('range_min', 0.01)
        range_max = chlor_a_config.get('range_max', 20.0)

        data_clipped = np.clip(data, range_min, range_max)
        data_normalized = (data_clipped - range_min) / (range_max - range_min)
        palette_min, palette_max = palette_values[0], palette_values[-1]
        data_for_palette = data_normalized * (palette_max - palette_min) + palette_min

    else:
        # Hirata palettes: direct value mapping
        data_for_palette = data

    get_color_vectorized = create_vectorized_color_ramp(palette_values, palette_colors, verbose)
    rgb_array_hwc = get_color_vectorized(data_for_palette)

    # Convert from (H, W, C) to (C, H, W) format
    height, width = data.shape
    rgb_array = np.zeros((3, height, width), dtype=np.uint8)
    rgb_array[0] = rgb_array_hwc[:, :, 0]  # Red
    rgb_array[1] = rgb_array_hwc[:, :, 1]  # Green
    rgb_array[2] = rgb_array_hwc[:, :, 2]  # Blue

    return rgb_array

def export_rgb(date_dir_path: str, config: Dict, masks_dict: Dict, input_dir_path: str, output_dir_path: str, verbose: bool) -> None:
    """Export RGB composite image."""
    # Input paths
    r_path = os.path.join(input_dir_path, config['nc_to_gtiff']['nc_exports']['rhos_655']['save_name'])
    g_path = os.path.join(input_dir_path, config['nc_to_gtiff']['nc_exports']['rhos_561']['save_name'])
    b_path = os.path.join(input_dir_path, config['nc_to_gtiff']['nc_exports']['rhos_482']['save_name'])

    # Read RGB bands
    with rasterio.open(r_path) as r:
        red = r.read(1)
        profile = r.profile

    with rasterio.open(g_path) as g:
        green = g.read(1)

    with rasterio.open(b_path) as b:
        blue = b.read(1)

    # Process RGB
    bands = np.stack((red, green, blue), axis=0)
    rgb = np.log(bands/0.01) / np.log(1/0.01)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = np.clip(rgb, 0.0, 0.8)
    rgb *= (1.0/rgb.max())

    # Convert to uint8
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    # Write RGBA image for rgb
    rgb_output_path = os.path.join(output_dir_path, config['palette_map']['rgb']['exports']['rgb']['save_name'])
    rgb_alpha_mask_name = config['palette_map']['rgb']['exports']['rgb']['alpha_mask']
    rgb_alpha_255 = make_alpha_band_from_mask(date_dir_path, config, masks_dict, rgb_alpha_mask_name, verbose)
    write_rgba_image(rgb_output_path, rgb_uint8, rgb_alpha_255, profile, verbose)

    # Write RGBA image for land clouds
    land_clouds_output_path = os.path.join(output_dir_path, config['palette_map']['rgb']['exports']['land_clouds']['save_name'])
    land_clouds_alpha_mask_name = config['palette_map']['rgb']['exports']['land_clouds']['alpha_mask']
    land_clouds_alpha_255 = make_alpha_band_from_mask(date_dir_path, config, masks_dict, land_clouds_alpha_mask_name, verbose)
    write_rgba_image(land_clouds_output_path, rgb_uint8, land_clouds_alpha_255, profile, verbose)

def export_chlor_a(date_dir_path: str, config: Dict, masks_dict: Dict, input_dir_path: str, output_dir_path: str, verbose: bool) -> None:
    """Export chlorophyll-a images (oceancolor and grayscale)."""
    input_path = os.path.join(input_dir_path, config['nc_to_gtiff']['nc_exports']['chlor_a']['save_name'])
    chlor_a_config = config['palette_map']['chlor_a']

    # Read input data
    with rasterio.open(input_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    # Create alpha band
    alpha_mask_name = chlor_a_config['alpha_mask']
    alpha_255 = make_alpha_band_from_mask(date_dir_path, config, masks_dict, alpha_mask_name, verbose)

    # Export each variant
    for export_name, export_config in chlor_a_config['exports'].items():
        output_path = os.path.join(output_dir_path, export_config['save_name'])
        palette_name = export_config['palette']
        palette = config['palette_map']['palettes'][palette_name]

        rgb_array = apply_palette_to_data(data, {'palette': palette}, palette_name, chlor_a_config, verbose)
        write_rgba_image(output_path, rgb_array, alpha_255, profile, verbose)


def export_hirata(date_dir_path: str, config: Dict, masks_dict: Dict, input_dir_path: str, output_dir_path: str, verbose: bool) -> None:
    """Export all Hirata palette images."""
    hirata_config = config['palette_map']['hirata']
    alpha_mask_name = hirata_config['alpha_mask']

    # Create alpha band once for all Hirata exports
    alpha_255 = make_alpha_band_from_mask(date_dir_path, config, masks_dict, alpha_mask_name, verbose)

    for hirata_type, export_config in hirata_config['exports'].items():
        input_path = os.path.join(input_dir_path, config['nc_to_gtiff']['nc_exports'][hirata_type]['save_name'])
        output_path = os.path.join(output_dir_path, export_config['save_name'])
        palette_name = export_config['palette']
        palette = config['palette_map']['palettes'][palette_name]

        # Read input data
        with rasterio.open(input_path) as src:
            data = src.read(1).astype(np.float32)
            profile = src.profile.copy()

        # Apply palette
        rgb_array = apply_palette_to_data(data, {'palette': palette}, palette_name, {}, verbose)

        # Write RGBA image
        write_rgba_image(output_path, rgb_array, alpha_255, profile, verbose)

def palette_map(config: Dict, date_dir_path: str) -> None:
    """Export all colorized images."""
    verbose = config['verbose']['palette_map']

    # Create output directory
    output_dir = os.path.join(date_dir_path, 'pipeline/palette_map')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    masks_dict = process_masks(date_dir_path, config, verbose)

    if config['palette_map']['processing_products']['rotated']:
        input_dir_rotated = os.path.join(date_dir_path, config['paths']['rotation_out_rel_path'], config['rotation']['nc_out_rel_path'])
        output_dir_rotated = os.path.join(output_dir, "rotated")
        os.makedirs(output_dir_rotated)
        export_rgb(date_dir_path, config, masks_dict, input_dir_rotated, output_dir_rotated, verbose)
        export_chlor_a(date_dir_path, config, masks_dict, input_dir_rotated, output_dir_rotated, verbose)
        export_hirata(date_dir_path, config, masks_dict, input_dir_rotated, output_dir_rotated, verbose)
    if config['palette_map']['processing_products']['interpolated']:
        input_dir_interpolated = os.path.join(date_dir_path, config['paths']['interpolation_out_rel_path'])
        output_dir_interpolated = os.path.join(output_dir, "interpolated")
        os.makedirs(output_dir_interpolated)
        export_chlor_a(date_dir_path, config, masks_dict, input_dir_interpolated, output_dir_interpolated, verbose)
        export_hirata(date_dir_path, config, masks_dict, input_dir_interpolated, output_dir_interpolated, verbose)
    if config['palette_map']['processing_products']['destriped']:
        input_dirs_stripe_corrected = (os.path.join(date_dir_path, config['paths']['stripe_correction_out_rel_path'], key) for key in config['stripe_correction'].keys())
        output_dir_destriped = os.path.join(output_dir, "destriped")
        os.makedirs(output_dir_destriped)
        export_rgb(date_dir_path, config, masks_dict, os.path.join(date_dir_path, config['paths']['stripe_correction_out_rel_path'], "land_cloud_reqs"), output_dir_destriped, verbose)
        export_chlor_a(date_dir_path, config, masks_dict, os.path.join(date_dir_path, config['paths']['stripe_correction_out_rel_path'], "chlor_a"), output_dir_destriped, verbose)
        export_hirata(date_dir_path, config, masks_dict, os.path.join(date_dir_path, config['paths']['stripe_correction_out_rel_path'], "hirata"), output_dir_destriped, verbose)
