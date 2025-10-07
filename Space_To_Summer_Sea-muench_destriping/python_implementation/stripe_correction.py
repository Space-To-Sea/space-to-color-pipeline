import os
import shutil

from typing import Tuple, Dict
import cv2
import numpy as np
import pywt
from scipy.fft import fft, ifft, fftshift, ifftshift
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine
from skimage import exposure
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

import numpy as np

def muench(image: np.ndarray, nodata: float, dec_num_v: int,
           sigma_v: float, dec_num_h: int, sigma_h: float,
           wavelet: str = 'db25', collect_stats: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Remove vertical stripes using Muench et al. (2009) method.

    Args:
        image: Input image
        nodata: No data value
        dec_num_v: Number of vertical decomposition levels
        sigma_v: Gaussian damping parameter for vertical
        dec_num_h: Number of horizontal decomposition levels
        sigma_h: Gaussian damping parameter for horizontal
        wavelet: Wavelet type
        collect_stats: Whether to collect statistics for debugging

    Returns:
        Tuple of (processed_image, statistics_dict)
    """
    input_dtype = image.dtype
    ima = image.astype(np.float64)
    if nodata is not None:
        ima[ima == nodata] = 0

    data_min = np.min(ima)
    data_max = np.max(ima)

    print(f"{data_min=}")
    print(f"{data_max=}")

    dec_num = max(dec_num_v, dec_num_h)

    # Store detail coefficients
    Ch, Cv, Cd = [], [], []
    Ch_stripes, Cv_stripes, Cd_stripes = [], [], []

    # Initialize statistics collection if requested
    statistics = {}
    if collect_stats:
        statistics = {
            "ima_levels": [None] * dec_num,
            "nima_levels": [None] * dec_num,
            "level_stripes": [None] * dec_num,
            "total_stripes": [None] * dec_num,
        }

    # Wavelet decomposition
    for i in range(dec_num):
        if collect_stats:
            statistics["ima_levels"][i] = ima.copy()

        ima, (ch, cv, cd) = pywt.dwt2(ima, wavelet)
        Ch.append(ch)
        Cv.append(cv)
        Cd.append(cd)

        if collect_stats:
            Ch_stripes.append(ch.copy())
            Cv_stripes.append(cv.copy())
            Cd_stripes.append(cd.copy())

    # Dampen vertical stripes
    for i in range(dec_num_v):
        fCv = fftshift(fft(Cv[i], axis=0), axes=0)
        my, mx = fCv.shape

        # Create damping function for vertical stripes
        freq_indices = np.arange(-my//2, -my//2 + my)
        damp = 1 - np.exp(-(freq_indices**2) / (2 * sigma_v**2))

        # Apply damping
        fCv_damped = fCv * damp[:, np.newaxis]

        # Inverse FFT
        damped_cv = np.real(ifft(ifftshift(fCv_damped, axes=0), axis=0))
        Cv[i] = damped_cv

        if collect_stats:
            Cv_stripes[i] -= damped_cv

    # Dampen horizontal stripes
    for i in range(dec_num_h):
        fCh = fftshift(fft(Ch[i], axis=1), axes=1)
        my, mx = fCh.shape

        # Create damping function for horizontal stripes
        freq_indices = np.arange(-mx//2, -mx//2 + mx)
        damp = 1 - np.exp(-(freq_indices**2) / (2 * sigma_h**2))

        # Apply damping along horizontal axis
        fCh_damped = fCh * damp[np.newaxis, :]

        # Inverse FFT
        damped_ch = np.real(ifft(ifftshift(fCh_damped, axes=1), axis=1))
        Ch[i] = damped_ch

        if collect_stats:
            Ch_stripes[i] -= damped_ch

    # Reconstruction
    nima = ima
    stripes_ima = None

    for i in range(max(dec_num_v, dec_num_h)-1, -1, -1):
        # Ensure dimensions match for reconstruction
        assert Ch[i].shape == Cv[i].shape and Cv[i].shape == Cd[i].shape
        target_shape = Ch[i].shape
        if nima.shape != target_shape:
            nima = nima[:target_shape[0], :target_shape[1]]
            if collect_stats and stripes_ima is not None:
                stripes_ima = stripes_ima[:target_shape[0], :target_shape[1]]

        nima = pywt.idwt2((nima, (Ch[i], Cv[i], Cd[i])), wavelet)

        if collect_stats:
            statistics["nima_levels"][i] = nima.copy()
            statistics["level_stripes"][i] = pywt.idwt2((None, (Ch_stripes[i], Cv_stripes[i], Cd_stripes[i])), wavelet)
            stripes_ima = pywt.idwt2((stripes_ima, (Ch_stripes[i], Cv_stripes[i], Cd_stripes[i])), wavelet)
            statistics["total_stripes"][i] = stripes_ima.copy() if stripes_ima is not None else None

    # Convert back to original dtype
    if input_dtype == np.uint8:
        nima = np.clip(nima, 0, 255).astype(np.uint8)
    elif input_dtype.kind == 'f':  # float types
        range_buffer = (data_max - data_min) * 0.1
        clip_min = max(data_min - range_buffer, 0.0)
        clip_max = data_max + range_buffer
        nima = np.clip(nima, clip_min, clip_max).astype(input_dtype)
    else:
        # For integer types, use original range
        if hasattr(np, 'iinfo'):
            try:
                info = np.iinfo(input_dtype)
                nima = np.clip(nima, max(info.min, data_min), min(info.max, data_max))
            except:
                nima = np.clip(nima, data_min, data_max)
        else:
            nima = np.clip(nima, data_min, data_max)
        nima = nima.astype(input_dtype)

    return nima.astype(input_dtype), statistics

def write_geotiff(data: np.ndarray, profile: rasterio.profiles.Profile, output_path: str):
    profile = profile.copy()
    profile.update(count=1, dtype=data.dtype)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)

def save_debug_geotiffs(statistics: Dict, profile: rasterio.profiles.Profile, debug_dir: str, filename_base: str):
    """Save debug GeoTIFFs for stripe analysis."""
    os.makedirs(debug_dir, exist_ok=True)

    # Save level stripes (stripes removed at each decomposition level)
    for i, level_stripes in enumerate(statistics.get("level_stripes", [])):
        if level_stripes is not None:
            debug_path = os.path.join(debug_dir, f"{filename_base}_level_{i}_stripes.tif")
            debug_profile = profile.copy()
            debug_profile.update(dtype=np.float64)

            with rasterio.open(debug_path, 'w', **debug_profile) as dst:
                dst.write(level_stripes, 1)

    # Save total stripes (cumulative stripes removed)
    total_stripes = statistics.get("total_stripes", [None])[0]
    if total_stripes is not None:
        debug_path = os.path.join(debug_dir, f"{filename_base}_total_stripes.tif")
        debug_profile = profile.copy()
        debug_profile.update(dtype=np.float64)

        with rasterio.open(debug_path, 'w', **debug_profile) as dst:
            dst.write(total_stripes, 1)

def save_histograms(statistics: Dict, histograms_dir: str, filename_base: str):
    """Save histograms of pixel values at different decomposition levels."""
    os.makedirs(histograms_dir, exist_ok=True)

    ima_levels = statistics.get("ima_levels", [])
    nima_levels = statistics.get("nima_levels", [])

    if not ima_levels and not nima_levels:
        return

    n_levels = len([x for x in ima_levels if x is not None])
    if n_levels == 0:
        return

    # fig, axes = plt.subplots(2, (n_levels + 1) // 2, figsize=(15, 8))
    # if n_levels == 1:
    #     axes = [axes]
    # elif n_levels <= 2:
    #     axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    # else:
    #     axes = axes.flatten()

    fig, axes = plt.subplots(2, (n_levels + 1) // 2, figsize=(15, 8))
    # Always flatten to ensure we have a 1D array of axes
    if hasattr(axes, 'flatten'):
        axes = axes.flatten()
    else:
        axes = [axes]  # Single axis case (shouldn't happen with our subplot call)

    for i in range(n_levels):
        if i < len(axes):
            ax = axes[i]

            # Plot IMA histogram
            if i < len(ima_levels) and ima_levels[i] is not None:
                ima_data = ima_levels[i].flatten()
                ax.hist(ima_data, bins=50, alpha=0.6, label='Original', color='blue', density=True)

            # Plot NIMA histogram
            if i < len(nima_levels) and nima_levels[i] is not None:
                nima_data = nima_levels[i].flatten()
                ax.hist(nima_data, bins=50, alpha=0.6, label='Processed', color='red', density=True)

            ax.set_title(f'Level {i} - Pixel Value Distribution')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for i in range(n_levels, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(histograms_dir, f'{filename_base}_histograms.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

def stripe_correction_helper(config: Dict, date_dir_path: str, config_key: str, input_path_key: str):
    verbose = config['verbose']['stripe_correction']

    # Create the main destriped directory path
    base_output_dir = os.path.join(date_dir_path, config['paths']['stripe_correction_out_rel_path'])

    # Create subdirectory based on config_key
    output_dir_path = os.path.join(base_output_dir, config_key)

    # Create debug directories if verbose
    if verbose:
        debug_dir = os.path.join(output_dir_path, "debug_geotiffs")
        histograms_dir = os.path.join(output_dir_path, "histograms")

    input_dir_path = os.path.join(date_dir_path, config['paths'][input_path_key])

    # Create the full directory path including subdirectory
    os.makedirs(output_dir_path, exist_ok=True)

    sc_config = config['stripe_correction'][config_key]

    apply_contrast_adjustment = sc_config['params']['apply_contrast_enhancement']
    dec_num_v = sc_config['params']['dec_num_v']
    dec_num_h = sc_config['params']['dec_num_h']
    sigma_v = sc_config['params']['sigma_v']
    sigma_h = sc_config['params']['sigma_h']
    wavelet_type = sc_config['params']['wavelet_type']

    for input_filename in sc_config['input_basenames']:
        # Handle different input path structures
        if input_path_key == 'rotation_out_rel_path':
            input_filepath = os.path.join(input_dir_path, config['rotation']['nc_out_rel_path'], input_filename)
        else:
            input_filepath = os.path.join(input_dir_path, input_filename)

        output_filepath = os.path.join(output_dir_path, input_filename)

        if not os.path.exists(input_filepath):
            print(f"Warning: Input file not found: {input_filepath}")
            continue

        with rasterio.open(input_filepath) as r:
            data = r.read(1)
            profile = r.profile
            nodata = r.nodata

        print(f"{input_filename=} and {nodata=} and {data.dtype=}")

        if apply_contrast_adjustment:
            valid_mask = (data != nodata)
            data[valid_mask] = exposure.equalize_adapthist(data[valid_mask])

        destriped_data, statistics = muench(
            data, nodata, dec_num_v, sigma_v, dec_num_h, sigma_h,
            wavelet_type, collect_stats=verbose
        )

        write_geotiff(destriped_data, profile, output_filepath)

        # Save debug outputs if verbose
        if verbose and statistics:
            filename_base = os.path.splitext(input_filename)[0]
            save_debug_geotiffs(statistics, profile, debug_dir, filename_base)
            save_histograms(statistics, histograms_dir, filename_base)

    return None

# def stripe_correction_setup(config: Dict, date_dir_path: str):
#     output_dir_path = os.path.join(date_dir_path, config['paths']['stripe_correction_out_rel_path'])
#     if os.path.exists(output_dir_path):
#         shutil.rmtree(output_dir_path, ignore_errors=True)
#     os.makedirs(output_dir_path)

def stripe_correction_land_cloud_reqs(config: Dict, date_dir_path: str):
    output_dir_path = os.path.join(date_dir_path, config['paths']['stripe_correction_out_rel_path'])
    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path, ignore_errors=True)
    os.makedirs(output_dir_path)
    return stripe_correction_helper(
        config, date_dir_path,
        config_key='land_cloud_reqs',
        input_path_key='rotation_out_rel_path'
    )

def stripe_correction_rrs_derived(config: Dict, date_dir_path: str):
    return stripe_correction_helper(
        config, date_dir_path,
        config_key='rrs_derived',
        input_path_key='interpolation_out_rel_path'
    )

def stripe_correction_chlor_a(config: Dict, date_dir_path: str):
    return stripe_correction_helper(
        config, date_dir_path,
        config_key='chlor_a',
        input_path_key='interpolation_out_rel_path'
    )

def stripe_correction_hirata(config: Dict, date_dir_path: str):
    return stripe_correction_helper(
        config, date_dir_path,
        config_key='hirata',
        input_path_key='interpolation_out_rel_path'
    )
