#!/usr/bin/env python

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

def muench_complete_modified(image: np.ndarray, nodata: np.ndarray, dec_num_v: int,
                               sigma_v: float, dec_num_h: int,
                               sigma_h: float, intermediates_dir_path: str, wavelet: str = 'db25') -> Tuple[np.ndarray, Dict]:
        """
        Remove vertical stripes using Muench et al. (2009) method.

        Args:
            image: Input image
            dec_num: Number of wavelet decomposition levels
            sigma: Gaussian damping parameter
            wavelet: Wavelet type

        Returns:
            Image with vertical stripes removed
        """
        assert image.dtype == np.uint8
        ima = image.astype(np.float64)
        if nodata is not None:
            ima[nodata == 1] = 0 #np.nan

        dec_num = max(dec_num_v, dec_num_h)

        # Store detail coefficients
        Ch, Cv, Cd = [], [], []
        Ch_stripes, Cv_stripes, Cd_stripes = [], [], []
        statistics = {
            "ima_levels": [None] * dec_num,
            "nima_levels": [None] * dec_num,
            "vertical_fourier_coeffs": [None] * dec_num,
            "vertical_fourier_frequencies": [None] * dec_num,
            "horizontal_fourier_coeffs": [None] * dec_num,
            "horizontal_fourier_frequencies": [None] * dec_num,
            "vertical_fourier_coeffs_damped": [None] * dec_num,
            "horizontal_fourier_coeffs_damped": [None] * dec_num,
            "level_stripes": [None] * dec_num,
            "total_stripes": [None] * dec_num,
        }
        # If I save imas images, average value and pixel histogram can be calculated from them at each dec_num and same with nimas
        # Append stripes components at the end

        # Wavelet decomposition
        for i in range(dec_num):
            ima, (ch, cv, cd) = pywt.dwt2(ima, wavelet)
            Ch.append(ch)
            Cv.append(cv)
            Cd.append(cd)
            Ch_stripes.append(ch.copy())
            Cv_stripes.append(cv.copy())
            Cd_stripes.append(cd.copy())
            statistics["ima_levels"][i] = ima.copy()


        # Dampen vertical stripes first
        for i in range(dec_num_v):
            fCv = fftshift(fft(Cv[i], axis=0), axes=0)
            my, mx = fCv.shape

            # Create damping function for vertical stripes
            # Gaussian damping in frequency domain
            freq_indices = np.arange(-my//2, -my//2 + my) # Creates a vector of frequencies that are in fCv
            frequencies = freq_indices / my
            statistics["vertical_fourier_frequencies"][i] = frequencies
            damp = 1 - np.exp(-(freq_indices**2) / (2 * sigma_v**2)) # Makes low frequencies lower and high frequencies higher by squaring them. low frequencies have e^-x closer to e^-0 = 1 so 1-e^... closer to 0. high frequencies have e^-x closer to -inf so e^inf which is higher. It increases the frequency of these.
            # Apply damping
            fCv_damped = fCv * damp[:, np.newaxis] # This takes a 2D fourier transform (x and y axis are both frequencies and pixel values are magnitudes) my by mx and multiplies it elementwise by an my by 1 vector which is broadcasted for each column.
            # Small σ → Narrow suppression, sharp cutoff
            # > Only suppresses frequencies very close to DC (zero frequency)
            # > Creates a steep high-pass filter
            # > More aggressive stripe removal but preserves more low-frequency image content
            # Large σ → Wide suppression, gentle cutoff
            # > Suppresses a broader range of low frequencies
            # > Creates a gradual high-pass filter
            # > Less aggressive stripe removal but may also remove legitimate low-frequency image features
            statistics["vertical_fourier_coeffs"][i] = fCv
            statistics["vertical_fourier_coeffs_damped"][i] = fCv_damped
            # Inverse FFT
            # Cv_temp = Cv[i].copy()
            damped_cv = np.real(ifft(ifftshift(fCv_damped, axes=0), axis=0))
            Cv[i] = damped_cv
            Cv_stripes[i] -= damped_cv

        # Dampen horizontal stripes
        for i in range(dec_num_h):
            fCh = fftshift(fft(Ch[i], axis=1), axes=1)
            my, mx = fCh.shape

            # Create damping function for horizontal stripes
            # Gaussian damping in frequency domain along horizontal axis
            freq_indices = np.arange(-mx//2, -mx//2 + mx)
            frequencies = freq_indices / mx
            statistics["horizontal_fourier_frequencies"][i] = frequencies
            damp = 1 - np.exp(-(freq_indices**2) / (2 * sigma_h**2))

            # Apply damping along horizontal axis
            fCh_damped = fCh * damp[np.newaxis, :]
            statistics["horizontal_fourier_coeffs"][i] = fCh
            statistics["horizontal_fourier_coeffs_damped"][i] = fCh_damped

            # Inverse FFT
            damped_ch = np.real(ifft(ifftshift(fCh_damped, axes=1), axis=1))
            Ch[i] = damped_ch
            Ch_stripes[i] -= damped_ch

        nima = ima
        stripes_ima = None
        for i in range(max(dec_num_v, dec_num_h)-1, -1, -1):
            # Ensure dimensions match for reconstruction
            assert Ch[i].shape == Cv[i].shape and Cv[i].shape == Cd[i].shape
            target_shape = Ch[i].shape
            if nima.shape != target_shape:
                nima = nima[:target_shape[0], :target_shape[1]]
                stripes_ima = stripes_ima[:target_shape[0], :target_shape[1]]

            nima = pywt.idwt2((nima, (Ch[i], Cv[i], Cd[i])), wavelet)
            statistics["nima_levels"][i] = nima.copy()
            statistics["level_stripes"][i] = pywt.idwt2((None, (Ch_stripes[i], Cv_stripes[i], Cd_stripes[i])), wavelet)
            stripes_ima = pywt.idwt2((stripes_ima, (Ch_stripes[i], Cv_stripes[i], Cd_stripes[i])), wavelet)
            statistics["total_stripes"][i] = stripes_ima.copy()

        assert nima.dtype == np.float64
        # nima[np.isnan(nima)] = 0
        nima = np.nan_to_num(nima, nan=0.0, posinf=255.0, neginf=0.0)
        nima = np.clip(nima, 0, 255).astype(np.uint8)

        return nima, statistics


def make_clean_dir(dir_path: str) -> None:
    if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def read_geotiff(filepath: str):
    """Read GeoTIFF preserving spatial reference."""
    with rasterio.open(filepath) as src:
        data = src.read()
        profile = src.profile.copy()

    assert data.shape[0] == 4
    assert data.dtype == np.uint8
    # RGBA - transpose to (H, W, C)
    return np.transpose(data, (1, 2, 0)), profile

def rotate_rgb_geotiff(input_path, output_path, angle=14.3):
    """
    Rotate a GeoTIFF image while maintaining georeferencing accuracy.

    Parameters:
    -----------
    input_path : str
        Path to input GeoTIFF file
    output_path : str
        Path to output rotated GeoTIFF file
    angle : float, default=14.3
        Angle of rotation in degrees
    shift_x : int, default=0
        Shift in x direction to adjust positioning
    shift_y : int, default=0
        Shift in y direction to adjust positioning
    adj_width : int, default=0
        Adjust width of output raster if needed
    adj_height : int, default=0
        Adjust height of output raster if needed

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

        # Create affine transformations for rotation and translation
        rotate = Affine.rotation(angle)

        # Combine affine transformations
        dst_transform = src_transform * rotate

        # Read all bands
        bands = []
        print(f"{src.count=}")
        for i in range(1, src.count + 1):
            print(f"Appending band {i}")
            bands.append(src.read(i))

        # Get dimensions and adjust if needed
        print(f"Number of bands: {len(bands)}")
        print(f"Shape of first band: {bands[0].shape}")
        height, width = bands[0].shape

        angle_rad = np.radians(angle)
        cos_a = abs(np.cos(angle_rad))
        sin_a = abs(np.sin(angle_rad))

        dst_width = int(np.ceil(width * cos_a + height * sin_a))
        dst_height = int(np.ceil(width * sin_a + height * cos_a))

        # And update the transform to center properly:
        center_x, center_y = width / 2.0, height / 2.0
        new_center_x, new_center_y = dst_width / 2.0, dst_height / 2.0

        #translate_to_center = Affine.translation(-center_x, -center_y)
        translate_to_center = Affine.translation(center_x, center_y)
        rotate_transform = Affine.rotation(angle)
        #translate_back = Affine.translation(0, -new_center_y)
        translate_back = Affine.translation(-new_center_x, -new_center_y)

        pixel_transform = translate_to_center * rotate_transform #* translate_back

        dst_transform = src_transform * pixel_transform * translate_back

        # Set properties for output file
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({
            "transform": dst_transform,
            "height": dst_height,
            "width": dst_width,
        })

        # Write rotated image to disk
        with rasterio.open(output_path, "w", **dst_kwargs) as dst:
            # Reproject each band
            for band_idx, band_data in enumerate(bands, 1):
                reproject(
                    source=band_data,
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src_transform,
                    src_crs=crs,
                    dst_transform=dst_transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest
                )

def write_geotiff_rgba(data: np.ndarray, alpha: np.ndarray, profile: rasterio.profiles.Profile,
                 output_path: str):
    """Write GeoTIFF preserving spatial reference."""
    assert data.ndim == 3
    # Clip data and alpha to the minimum common shape if they don't align
    # Ensure data and alpha have the same height and width by clipping or padding with zeros
    target_h, target_w = alpha.shape[:2]
    data_h, data_w = data.shape[:2]

    # Clip if data is larger
    clipped_data = data[:target_h, :target_w, ...]
    # Pad if data is smaller
    if clipped_data.shape[0] < target_h or clipped_data.shape[1] < target_w:
        pad_h = target_h - clipped_data.shape[0]
        pad_w = target_w - clipped_data.shape[1]
        pad_width = (
            (0, pad_h if pad_h > 0 else 0),
            (0, pad_w if pad_w > 0 else 0),
            (0, 0)
        )
        clipped_data = np.pad(clipped_data, pad_width, mode='constant', constant_values=0)
    data = clipped_data
    alpha = alpha[:target_h, :target_w]
    assert data.dtype == np.uint8
    assert alpha.dtype == np.uint8

    profile = profile.copy()

    # Update profile to include alpha band
    # profile.update(count=data.shape[2] + 1, dtype=np.uint8)

    # Combine RGB and alpha into single RGBA array
    rgba_data = np.concatenate([data, alpha[:,:,np.newaxis]], axis=2)

    # Transpose to (C, H, W) format for rasterio
    rgba_to_write = np.transpose(rgba_data, (2, 0, 1))

    with rasterio.open(output_path, 'w', **profile) as dst:
        # Write all 4 bands at once
        dst.write(rgba_to_write)

def write_geotiff(data: np.ndarray, profile: rasterio.profiles.Profile,
                 output_path: str):
    """Write GeoTIFF preserving spatial reference."""

    # assert data.dtype == np.uint8

    profile = profile.copy()
    profile.update(count=data.shape[2])

    data_to_write = np.transpose(data, (2, 0, 1))

    with rasterio.open(output_path, 'w', **profile) as dst:
        # Write all 4 bands at once
        dst.write(data_to_write)

def process_rgba(input_path: str, output_path: str,
            intermediates_dir: str, enhance_contrast: bool = False, dec_num_v=5, dec_num_h=5, sigma_v=2.4, sigma_h=2.4, save_statistics=True, min_max_rgba=()) -> None:
    """
    Process RGBA image using the same workflow as process_single_band.

    Args:
        input_path: Path to input RGBA GeoTIFF
        output_path: Path to save processed image
        intermediates_dir: Directory to store intermediate files
        enhance_contrast: Whether to apply adaptive histogram equalization
    """
    # Read image
    image, profile = read_geotiff(input_path)

    if image.ndim != 3:
        raise ValueError("RGB processing requires 3D image")

    if image.shape[2] != 4:
        raise ValueError("Expected RGBA image (4 bands), got {} bands".format(image.shape[2]))

    # Create intermediates directory
    os.makedirs(intermediates_dir, exist_ok=True)

    # Handle nodata using alpha channel
    processed_image, nodata_mask = image[:,:,:3], image[:,:,3]==0 #handle_nodata_rgba(image)

    # Enhance contrast if requested (only on RGB channels)
    if enhance_contrast:
        valid_mask = ~nodata_mask
        if np.any(valid_mask):
            for band_idx in range(3):  # Only RGB channels, not alpha
                processed_image[:, :, band_idx][valid_mask] = exposure.equalize_adapthist(
                    processed_image[:, :, band_idx][valid_mask].astype(np.uint8)
                ) * 255

    preprocessed_output_path = os.path.join(intermediates_dir, f"preprocessed.tif")
    preprocessed_alpha = np.where(nodata_mask, 0, 255).astype(np.uint8)
    write_geotiff_rgba(processed_image[:, :, :], preprocessed_alpha, profile.copy(), preprocessed_output_path)

    rotated_path = os.path.join(
        intermediates_dir, f"rotated.tif"
    )
    rotate_rgb_geotiff(
        preprocessed_output_path,
        rotated_path,
        angle=14.3
    )
    rotated_image, rotated_profile = read_geotiff(rotated_path)

    processed_bands = []
    all_statistics = []

    alpha = rotated_image[:,:,3]
    nodata = (alpha == 0)

    # Iterate over bands using the last axis (channels)
    for band_idx in range(3):
        band = rotated_image[:, :, band_idx]
        destriped_band, statistics = muench_complete_modified(
            band, nodata, dec_num_v, sigma_v, dec_num_h, sigma_h, intermediates_dir, wavelet='db25'
        )
        processed_bands.append(destriped_band)
        all_statistics.append(statistics)
    _, alpha_statistics = muench_complete_modified(
        alpha, None, dec_num_v, sigma_v, dec_num_h, sigma_h, intermediates_dir, wavelet='db25'
    )


    ###################################### Debugging with Alpha Band ###############################################################



    ################################################################################################################################

    processed_array = np.stack(processed_bands, axis=2)

    processed_output_path = os.path.join(intermediates_dir, f"processed.tif")
    write_geotiff_rgba(processed_array, alpha, rotated_profile.copy(), processed_output_path)

    if save_statistics:
        # Save histograms and create debug GeoTIFFs
        # save_histograms_and_debug_geotiffs(all_statistics, alpha_statistics, rotated_profile, intermediates_dir, dec_num_v, dec_num_h)
        save_histograms_and_debug_geotiffs_enhanced(all_statistics, alpha_statistics, rotated_profile, intermediates_dir, dec_num_v, dec_num_h)

def save_histograms_and_debug_geotiffs(all_statistics, alpha_statistics, profile, intermediates_dir, dec_num_v, dec_num_h):
    """
    Save histograms and create debug GeoTIFFs for each band.

    Args:
        all_statistics: List of statistics dictionaries from each RGB band
        alpha: Alpha channel array
        profile: Rasterio profile for GeoTIFF creation
        intermediates_dir: Directory to save files
        dec_num_v: Number of vertical decomposition levels
        dec_num_h: Number of horizontal decomposition levels
    """
    # Create directories for outputs
    histograms_dir = os.path.join(intermediates_dir, "histograms") # Histogram showing values of fourier coefficients for all dec_num pre and post damping; #Histogram showing values of all pixels pre and post damping for all dec_num (save pixel values for nima upon being rebuilt each time as well as pixel values for ima being decomposed each time;
    debug_geotiffs_dir = os.path.join(intermediates_dir, "debug_geotiffs") # One 4 band tiff per max(dec_num_v, dec_num_h) showing stripes removed for that dec_num + another image for total stripes removed from each band; should save as double right now cause want to keep negative values; should have profile recalculated;
    os.makedirs(histograms_dir, exist_ok=True)
    os.makedirs(debug_geotiffs_dir, exist_ok=True)

    band_names = ['red', 'green', 'blue', 'alpha']
    max_dec_levels = max(dec_num_v, dec_num_h)

    create_debug_geotiffs(all_statistics, alpha_statistics, profile, debug_geotiffs_dir, max_dec_levels)

    # Process RGB bands (indices 0, 1, 2)
    for band_idx in range(3):
        statistics = all_statistics[band_idx]
        band_name = band_names[band_idx]

        # Save histograms
        save_band_histograms(statistics, band_name, histograms_dir)

def save_band_histograms(statistics, band_name, histograms_dir):
    """Save histograms for decomposition levels and magnitude evolution."""

    # Save average values for each ima, nima for each max_dec_num and below save pixel histograms for each ima, nima for each max_dec_num
    # Save fourier coefficient histograms for each max_decnum pre damped and post damped below it

    # Save pixel histograms for each dec_num's ima (average values plotted above)

    # Save pixel histograms for each dec_num's nima (average values plotted above)

    # Save magnitude histograms for vertical fourier coefficients

    # Save magnitude histograms for vertical fourier coefficients after damping

    # Save magnitude histograms for horizontal fourier coefficients

    # Save magnitude histograms for horizontal fourier coefficients after damping


    # Save pixel histograms for decomposition levels
    save_band_image_histograms(band_name, statistics, histograms_dir, 'ima_levels', "config_set_range", (-100, 20000))
    save_band_image_histograms(band_name, statistics, histograms_dir, 'nima_levels', "config_set_range", (-100, 20000))

    # Save fourier coefficient magnitude histograms (original)
    save_band_fourier_coeff_histograms(band_name, statistics, histograms_dir, ("vertical_fourier_coeffs", "vertical_fourier_frequencies"))
    # save_band_fourier_coeff_histograms(band_name, statistics, histograms_dir, 'horizontal_fourier_coeffs')

    # Save fourier coefficient magnitude histograms (after damping)
    save_band_fourier_coeff_histograms(band_name, statistics, histograms_dir, ("vertical_fourier_coeffs_damped", "vertical_fourier_frequencies"))
    # save_band_fourier_coeff_histograms(band_name, statistics, histograms_dir, 'horizontal_fourier_coeffs_damped')

def save_band_image_histograms(band_name: str, statistics_dict: Dict, histograms_dir: str, statistics_key: str='ima_levels', mode: str = "auto_set_range", min_max: Tuple = None):
    image_levels = statistics_dict[statistics_key]
    image_histograms = []

    # Modes can be "auto_set_range", "min_max", or "config_set_range"

    if mode == "auto_set_range":
        set_range = None
    elif mode == "min_max":
        all_data = []
        for ima_level in image_levels:
            if ima_level is not None:
                all_data.extend(ima_level.flatten())
        min = np.min(all_data)
        max = np.max(all_data)
        set_range = (min, max)
    elif mode == "config_set_range":
        min = min_max[0]
        max = min_max[1]
        set_range = (min, max)


    for ima_level in image_levels:
        hist, bin_edges = np.histogram(ima_level, bins=50, range=set_range, density=False) #density=True)
        image_histograms.append({
            'histogram': hist,
            'bin_edges': bin_edges,
            'mean': np.mean(ima_level),
            # 'std': np.std(ima_level),
            'min': np.min(ima_level),
            'max': np.max(ima_level)
        })
    # Create subplots for all decomposition level histograms
    n_levels = len(image_levels)
    if n_levels > 0:
        fig, axes = plt.subplots(2, (n_levels + 1) // 2, figsize=(15, 8))
        if n_levels == 1:
            axes = [axes]
        elif n_levels <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, hist_data in enumerate(image_histograms):
            if i < len(axes):
                ax = axes[i]
                hist = hist_data['histogram']
                bin_edges = hist_data['bin_edges']
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                ax.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.7, edgecolor='black')
                ax.set_title(f'{band_name.title()} - {i}\n'
                            f'Mean: {hist_data["mean"]:.0f},\n Min: {hist_data["min"]:.1f},\n Max: {hist_data["max"]:.1f}')
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for i in range(n_levels, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(histograms_dir, f'{band_name}_{statistics_key}_histograms.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

# def save_band_fourier_coeff_histograms(band_name: str, statistics_dict: Dict, histograms_dir: str, statistics_key: str = "vertical_fourier_coeffs"):
#     """
#     Save histograms for Fourier coefficient magnitudes.

#     Args:
#         band_name: Name of the band (e.g., 'red', 'green', 'blue')
#         statistics_dict: Dictionary containing fourier coefficient data
#         histograms_dir: Directory to save histogram plots
#         statistics_key: Key for fourier coefficients ('vertical_fourier_coeffs' or 'horizontal_fourier_coeffs')
#     """


#     fourier_levels = statistics_dict[statistics_key]
#     histograms = []

#     # Create histograms for each decomposition level
#     for i, level in enumerate(fourier_levels):
#         if level is None:
#             histograms.append(None)
#         else:
#             # Get magnitude of complex fourier coefficients
#             magnitude = np.abs(level)
#             # Flatten the magnitude array for histogram
#             magnitude_flat = magnitude.flatten()

#             # Remove any NaN or infinite values
#             magnitude_flat = magnitude_flat[np.isfinite(magnitude_flat)]

#             if len(magnitude_flat) > 0:
#                 hist, bin_edges = np.histogram(magnitude_flat, bins=50, density=True)
#                 histograms.append({
#                     'histogram': hist,
#                     'bin_edges': bin_edges,
#                     'mean': np.mean(magnitude_flat),
#                     'std': np.std(magnitude_flat),
#                     'min': np.min(magnitude_flat),
#                     'max': np.max(magnitude_flat),
#                     'level': i
#                 })
#             else:
#                 histograms.append(None)

#     # Filter out None values for plotting
#     valid_histograms = [h for h in histograms if h is not None]
#     n_levels = len(valid_histograms)

#     if n_levels > 0:
#         # Determine subplot layout
#         cols = min(3, n_levels)
#         rows = (n_levels + cols - 1) // cols

#         fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
#         if n_levels == 1:
#             axes = [axes]
#         elif rows == 1:
#             axes = axes if isinstance(axes, np.ndarray) else [axes]
#         else:
#             axes = axes.flatten()

#         direction = statistics_key.split('_')[0]  # 'vertical' or 'horizontal'

#         for i, hist_data in enumerate(valid_histograms):
#             if i < len(axes):
#                 ax = axes[i]
#                 hist = hist_data['histogram']
#                 bin_edges = hist_data['bin_edges']
#                 bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#                 level = hist_data['level']

#                 ax.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.7, edgecolor='black')
#                 ax.set_title(f'{band_name.title()} - {direction.title()} FFT Level {level}\n'
#                             f'Mean: {hist_data["mean"]:.2e}, Std: {hist_data["std"]:.2e}')
#                 ax.set_xlabel('Magnitude')
#                 ax.set_ylabel('Density')
#                 ax.grid(True, alpha=0.3)

#                 # Use log scale for better visualization of fourier magnitudes
#                 ax.set_yscale('log')

#         # Hide extra subplots
#         for i in range(n_levels, len(axes)):
#             axes[i].set_visible(False)

#         plt.tight_layout()
#         plt.savefig(os.path.join(histograms_dir, f'{band_name}_{statistics_key}_magnitude_histograms.png'),
#                     dpi=150, bbox_inches='tight')
#         plt.close()

# def save_band_fourier_coeff_histograms(band_name: str, statistics_dict: Dict, histograms_dir: str, statistics_keys: Tuple = ("vertical_fourier_coeffs", "vertical_fourier_frequencies")):
#     """
#     Save histograms for Fourier coefficient magnitudes.

#     Args:
#         band_name: Name of the band (e.g., 'red', 'green', 'blue')
#         statistics_dict: Dictionary containing fourier coefficient data
#         histograms_dir: Directory to save histogram plots
#         statistics_key: Key for fourier coefficients ('vertical_fourier_coeffs' or 'horizontal_fourier_coeffs')
#     """


#     fourier_levels = statistics_dict[statistics_keys[0]]
#     fourier_frequencies = statistics_dict[statistics_keys[1]]
#     histograms = []

#     # # Create histograms for each decomposition level
#     # for i, level in enumerate(fourier_levels):
#     #     if level is None:
#     #         histograms.append(None)
#     #     else:
#     #         # Get magnitude of complex fourier coefficients
#     #         magnitude = np.abs(level)
#     #         # Flatten the magnitude array for histogram
#     #         magnitude_flat = magnitude.flatten()

#     #         # Remove any NaN or infinite values
#     #         magnitude_flat = magnitude_flat[np.isfinite(magnitude_flat)]

#     #         if len(magnitude_flat) > 0:
#     #             hist, bin_edges = np.histogram(magnitude_flat, bins=50, density=True)
#     #             histograms.append({
#     #                 'histogram': hist,
#     #                 'bin_edges': bin_edges,
#     #                 'mean': np.mean(magnitude_flat),
#     #                 'std': np.std(magnitude_flat),
#     #                 'min': np.min(magnitude_flat),
#     #                 'max': np.max(magnitude_flat),
#     #                 'level': i
#     #             })
#     #         else:
#     #             histograms.append(None)

#     # # Filter out None values for plotting
#     # valid_histograms = [h for h in histograms if h is not None]
#     # n_levels = len(valid_histograms)

#     # if n_levels > 0:
#     #     # Determine subplot layout
#     #     cols = min(3, n_levels)
#     #     rows = (n_levels + cols - 1) // cols

#     #     fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
#     #     if n_levels == 1:
#     #         axes = [axes]
#     #     elif rows == 1:
#     #         axes = axes if isinstance(axes, np.ndarray) else [axes]
#     #     else:
#     #         axes = axes.flatten()

#     #     direction = statistics_key.split('_')[0]  # 'vertical' or 'horizontal'

#     #     for i, hist_data in enumerate(valid_histograms):
#     #         if i < len(axes):
#     #             ax = axes[i]
#     #             hist = hist_data['histogram']
#     #             bin_edges = hist_data['bin_edges']
#     #             bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     #             level = hist_data['level']

#     #             ax.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.7, edgecolor='black')
#     #             ax.set_title(f'{band_name.title()} - {direction.title()} FFT Level {level}\n'
#     #                         f'Mean: {hist_data["mean"]:.2e}, Std: {hist_data["std"]:.2e}')
#     #             ax.set_xlabel('Magnitude')
#     #             ax.set_ylabel('Density')
#     #             ax.grid(True, alpha=0.3)

#     #             # Use log scale for better visualization of fourier magnitudes
#     #             ax.set_yscale('log')

#     #     # Hide extra subplots
#     #     for i in range(n_levels, len(axes)):
#     #         axes[i].set_visible(False)

#     #     plt.tight_layout()
#     #     plt.savefig(os.path.join(histograms_dir, f'{band_name}_{statistics_key}_magnitude_histograms.png'),
#     #                 dpi=150, bbox_inches='tight')
#     #     plt.close()

#     histograms = []
#     freq_magnitude_data = []

#     # Create histograms for each decomposition level
#     for i, (level, frequencies) in enumerate(zip(fourier_levels, fourier_frequencies)):
#         if level is None or frequencies is None:
#             histograms.append(None)
#             freq_magnitude_data.append(None)
#         else:
#             # Get magnitude of complex fourier coefficients
#             magnitude = np.abs(level)

#             # For histogram: flatten the magnitude array
#             magnitude_flat = magnitude.flatten()
#             magnitude_flat = magnitude_flat[np.isfinite(magnitude_flat)]

#             if len(magnitude_flat) > 0:
#                 hist, bin_edges = np.histogram(magnitude_flat, bins=50, density=True)
#                 histograms.append({
#                     'histogram': hist,
#                     'bin_edges': bin_edges,
#                     'mean': np.mean(magnitude_flat),
#                     'std': np.std(magnitude_flat),
#                     'min': np.min(magnitude_flat),
#                     'max': np.max(magnitude_flat),
#                     'level': i
#                 })

#                 # For frequency vs magnitude plot: average magnitude across perpendicular axis
#                 if coeffs_key.startswith('vertical'):
#                     # Vertical FFT: average across horizontal dimension (axis=1)
#                     magnitude_profile = np.mean(magnitude, axis=1)
#                 else:  # horizontal
#                     # Horizontal FFT: average across vertical dimension (axis=0)
#                     magnitude_profile = np.mean(magnitude, axis=0)

#                 freq_magnitude_data.append({
#                     'frequencies': frequencies,
#                     'magnitude_profile': magnitude_profile,
#                     'level': i
#                 })
#             else:
#                 histograms.append(None)
#                 freq_magnitude_data.append(None)

#     # # Create magnitude histograms (existing functionality)
#     # valid_histograms = [h for h in histograms if h is not None]
#     # n_hist_levels = len(valid_histograms)

#     # if n_hist_levels > 0:
#     #     # Determine subplot layout for histograms
#     #     cols = min(3, n_hist_levels)
#     #     rows = (n_hist_levels + cols - 1) // cols

#     #     fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
#     #     if n_hist_levels == 1:
#     #         axes = [axes]
#     #     elif rows == 1:
#     #         axes = axes if isinstance(axes, np.ndarray) else [axes]
#     #     else:
#     #         axes = axes.flatten()

#     #     direction = coeffs_key.split('_')[0]  # 'vertical' or 'horizontal'

#     #     for i, hist_data in enumerate(valid_histograms):
#     #         if i < len(axes):
#     #             ax = axes[i]
#     #             hist = hist_data['histogram']
#     #             bin_edges = hist_data['bin_edges']
#     #             bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     #             level = hist_data['level']

#     #             ax.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.7, edgecolor='black')
#     #             ax.set_title(f'{band_name.title()} - {direction.title()} FFT Level {level}\n'
#     #                         f'Mean: {hist_data["mean"]:.2e}, Std: {hist_data["std"]:.2e}')
#     #             ax.set_xlabel('Magnitude')
#     #             ax.set_ylabel('Density')
#     #             ax.grid(True, alpha=0.3)
#     #             ax.set_yscale('log')

#     #     # Hide extra subplots
#     #     for i in range(n_hist_levels, len(axes)):
#     #         axes[i].set_visible(False)

#     #     plt.tight_layout()
#     #     plt.savefig(os.path.join(histograms_dir, f'{band_name}_{coeffs_key}_magnitude_histograms.png'),
#     #                 dpi=150, bbox_inches='tight')
#     #     plt.close()

#     # Create frequency vs magnitude plots (NEW functionality)
#     valid_freq_data = [d for d in freq_magnitude_data if d is not None]
#     n_freq_levels = len(valid_freq_data)

#     if n_freq_levels > 0:
#         # Determine subplot layout for frequency plots
#         cols = min(3, n_freq_levels)
#         rows = (n_freq_levels + cols - 1) // cols

#         fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
#         if n_freq_levels == 1:
#             axes = [axes]
#         elif rows == 1:
#             axes = axes if isinstance(axes, np.ndarray) else [axes]
#         else:
#             axes = axes.flatten()

#         direction = coeffs_key.split('_')[0]  # 'vertical' or 'horizontal'

#         for i, freq_data in enumerate(valid_freq_data):
#             if i < len(axes):
#                 ax = axes[i]
#                 frequencies = freq_data['frequencies']
#                 magnitude_profile = freq_data['magnitude_profile']
#                 level = freq_data['level']

#                 # Plot magnitude vs frequency
#                 ax.plot(frequencies, magnitude_profile, 'b-', alpha=0.8, linewidth=1.5)
#                 ax.set_xlabel(f'{direction.title()} Frequency (cycles/pixel)')
#                 ax.set_ylabel('Magnitude')
#                 ax.set_title(f'{band_name.title()} - {direction.title()} FFT Level {level}\nMagnitude vs Frequency')
#                 ax.grid(True, alpha=0.3)
#                 ax.set_yscale('log')

#                 # Highlight DC component (zero frequency)
#                 ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='DC (0 Hz)')
#                 ax.legend()

#                 # Set x-axis limits to show the full frequency range
#                 ax.set_xlim(frequencies.min(), frequencies.max())

#         # Hide extra subplots
#         for i in range(n_freq_levels, len(axes)):
#             axes[i].set_visible(False)

#         plt.tight_layout()
#         plt.savefig(os.path.join(histograms_dir, f'{band_name}_{coeffs_key}_frequency_vs_magnitude.png'),
#                     dpi=150, bbox_inches='tight')
#         plt.close()

def save_band_fourier_coeff_histograms(
    band_name: str,
    statistics_dict: Dict,
    histograms_dir: str,
    statistics_keys: tuple = ("vertical_fourier_coeffs", "vertical_fourier_frequencies"),
):
    """
    Save histograms for Fourier coefficient magnitudes and frequency vs magnitude plots.

    Args:
        band_name: Name of the band (e.g., 'red', 'green', 'blue')
        statistics_dict: Dictionary containing fourier coefficient data
        histograms_dir: Directory to save histogram plots
        statistics_keys: Tuple of keys for coefficients and frequencies
    """
    coeffs_key, freqs_key = statistics_keys
    fourier_levels = statistics_dict[coeffs_key]
    fourier_frequencies = statistics_dict[freqs_key]

    histograms = []
    freq_magnitude_data = []

    # Create histograms and frequency-magnitude data for each decomposition level
    for i, (level, frequencies) in enumerate(zip(fourier_levels, fourier_frequencies)):
        if level is None or frequencies is None:
            histograms.append(None)
            freq_magnitude_data.append(None)
            continue

        magnitude = np.abs(level)
        magnitude_flat = magnitude.flatten()
        magnitude_flat = magnitude_flat[np.isfinite(magnitude_flat)]

        if len(magnitude_flat) > 0:
            hist, bin_edges = np.histogram(magnitude_flat, bins=50, density=True)
            histograms.append({
                'histogram': hist,
                'bin_edges': bin_edges,
                'mean': np.mean(magnitude_flat),
                'std': np.std(magnitude_flat),
                'min': np.min(magnitude_flat),
                'max': np.max(magnitude_flat),
                'level': i
            })

            # For frequency vs magnitude plot: average magnitude across perpendicular axis
            if coeffs_key.startswith('vertical'):
                # Vertical FFT: average across horizontal dimension (axis=1)
                magnitude_profile = np.mean(magnitude, axis=1)
            else:
                # Horizontal FFT: average across vertical dimension (axis=0)
                magnitude_profile = np.mean(magnitude, axis=0)

            freq_magnitude_data.append({
                'frequencies': frequencies,
                'magnitude_profile': magnitude_profile,
                'level': i
            })
        else:
            histograms.append(None)
            freq_magnitude_data.append(None)

    # # Plot magnitude histograms
    # valid_histograms = [h for h in histograms if h is not None]
    # n_hist_levels = len(valid_histograms)
    # if n_hist_levels > 0:
    #     cols = min(3, n_hist_levels)
    #     rows = (n_hist_levels + cols - 1) // cols
    #     fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    #     if n_hist_levels == 1:
    #         axes = [axes]
    #     elif rows == 1:
    #         axes = axes if isinstance(axes, np.ndarray) else [axes]
    #     else:
    #         axes = axes.flatten()
    #     direction = coeffs_key.split('_')[0]  # 'vertical' or 'horizontal'
    #     for i, hist_data in enumerate(valid_histograms):
    #         if i < len(axes):
    #             ax = axes[i]
    #             hist = hist_data['histogram']
    #             bin_edges = hist_data['bin_edges']
    #             bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #             level = hist_data['level']
    #             ax.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.7, edgecolor='black')
    #             ax.set_title(f'{band_name.title()} - {direction.title()} FFT Level {level}\n'
    #                          f'Mean: {hist_data["mean"]:.2e}, Std: {hist_data["std"]:.2e}')
    #             ax.set_xlabel('Magnitude')
    #             ax.set_ylabel('Density')
    #             ax.grid(True, alpha=0.3)
    #             ax.set_yscale('log')
    #     for i in range(n_hist_levels, len(axes)):
    #         axes[i].set_visible(False)
    #     plt.tight_layout()
    #     plt.savefig(
    #         os.path.join(histograms_dir, f'{band_name}_{coeffs_key}_magnitude_histograms.png'),
    #         dpi=150, bbox_inches='tight'
    #     )
    #     plt.close()

    # Plot frequency vs magnitude
    valid_freq_data = [d for d in freq_magnitude_data if d is not None]
    n_freq_levels = len(valid_freq_data)
    if n_freq_levels > 0:
        cols = min(3, n_freq_levels)
        rows = (n_freq_levels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_freq_levels == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        direction = coeffs_key.split('_')[0]
        for i, freq_data in enumerate(valid_freq_data):
            if i < len(axes):
                ax = axes[i]
                frequencies = freq_data['frequencies']
                magnitude_profile = freq_data['magnitude_profile']
                level = freq_data['level']
                ax.plot(frequencies, magnitude_profile, 'b-', alpha=0.8, linewidth=1.5)
                ax.set_xlabel(f'{direction.title()} Frequency (cycles/pixel)')
                ax.set_ylabel('Magnitude')
                ax.set_title(f'{band_name.title()} - {direction.title()} FFT Level {level}\nMagnitude vs Frequency')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='DC (0 Hz)')
                ax.legend()
                ax.set_xlim(frequencies.min(), frequencies.max())
        for i in range(n_freq_levels, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(histograms_dir, f'{band_name}_{coeffs_key}_frequency_vs_magnitude.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()


def create_debug_geotiffs(all_statistics, alpha_statistics, profile, debug_geotiffs_dir, max_dec_levels):
    """Create a debug GeoTIFF with wavelet level stripes and complete stripes."""

    # Create geotiffs for each level of stripes, a geotiff for the final stripes shown. Each geotiff is rgba for each band.

    # Also save geotiffs for each level of imas and nimas with imas and nimas being paired together.

    ima_nima_float64_level_arrays = [[None]*6 for _ in range(max_dec_levels)] # Ends up as a list with element lists for each dec level.
    for i, band_statistics in enumerate(all_statistics):
        nima_float64_levels = band_statistics["nima_levels"]
        ima_float64_levels = band_statistics["ima_levels"]
        for j in range(max_dec_levels):
            ima_nima_float64_level_arrays[j][i] = ima_float64_levels[j]
            ima_nima_float64_level_arrays[j][i+3] = nima_float64_levels[j] # ima_r, nima_r, ima_g, nima_g, ima_b, nima_b
    for dec_level, band_ima_nima in enumerate(ima_nima_float64_level_arrays):
        '[(r_ima, r_nima), (g_ima, g_nima), (b_ima, b_nima)]'
        debug_path = os.path.join(debug_geotiffs_dir, f"dn_{dec_level}_rgb_ima_nima_float64.tif")
        debug_profile = profile.copy()
        debug_profile.update(count=6)
        stacked_arrays = stack_jagged_matrices(band_ima_nima)
        write_geotiff(stacked_arrays, debug_profile, debug_path) # Need debug_profile

    level_stripes_float64_arrays = [[] for _ in range(max_dec_levels)] # Ends up as a list with element lists for each dec level.
    for _, band_statistics in enumerate(all_statistics):
        level_stripes_float64 = band_statistics["level_stripes"]
        for j in range(max_dec_levels):
            level_stripes_float64_arrays[j].append(level_stripes_float64[j]) # level stripes for each level are in rgb order
    alpha_level_stripes_float64 = alpha_statistics["level_stripes"]
    for j in range(max_dec_levels):
        level_stripes_float64_arrays[j].append(alpha_level_stripes_float64[j])
    for dec_level, band_level_stripes in enumerate(level_stripes_float64_arrays):
        debug_path = os.path.join(debug_geotiffs_dir, f"dn_{dec_level}_rgb_level_stripes_float64.tif")
        debug_profile = profile.copy()
        debug_profile.update(count=4)
        stacked_arrays = stack_jagged_matrices(band_level_stripes)
        write_geotiff(stacked_arrays, debug_profile, debug_path) # Need debug_profile

    total_stripes_float64 = []
    for _, band_statistics in enumerate(all_statistics):
        total_stripes_float64.append(band_statistics["total_stripes"][0])
    alpha_total_stripes_float64 = alpha_statistics["total_stripes"][0]
    total_stripes_float64.append(alpha_total_stripes_float64)
    debug_path = os.path.join(debug_geotiffs_dir, f"rgb_total_stripes_float64.tif")
    debug_profile = profile.copy()
    debug_profile.update(count=4)
    stacked_arrays = stack_jagged_matrices(total_stripes_float64)
    write_geotiff(stacked_arrays, debug_profile, debug_path) # Need debug_profile

def stack_jagged_matrices(matrices_list):
    """Properly stack matrices of different sizes by padding with NaN."""
    if not matrices_list:
        return np.array([])

    # Find maximum dimensions
    max_rows = max(matrix.shape[0] for matrix in matrices_list)
    max_cols = max(matrix.shape[1] for matrix in matrices_list)

    # Pad each matrix to the maximum size
    padded_matrices = []
    for matrix in matrices_list:
        padded = np.full((max_rows, max_cols), 0, dtype=np.float64)
        padded[:matrix.shape[0], :matrix.shape[1]] = matrix
        padded_matrices.append(padded)

    return np.stack(padded_matrices, axis=2)

def save_band_image_histograms_combined(band_name: str, statistics_dict: Dict, histograms_dir: str,
                                       mode: str = "auto_set_range", min_max: Tuple = (-100, 20000)):
    """
    Save combined histograms for both ima_levels and nima_levels on the same plots.

    Args:
        band_name: Name of the band (e.g., 'red', 'green', 'blue')
        statistics_dict: Dictionary containing image level data
        histograms_dir: Directory to save histogram plots
        mode: Histogram range mode ("auto_set_range", "min_max", or "config_set_range")
        min_max: Min/max values when using "config_set_range" mode
    """
    ima_levels = statistics_dict['ima_levels']
    nima_levels = statistics_dict['nima_levels']

    # Determine range based on mode
    if mode == "auto_set_range":
        set_range = None
    elif mode == "min_max":
        all_data = []
        for level in ima_levels + nima_levels:
            if level is not None:
                all_data.extend(level.flatten())
        min_val, max_val = np.min(all_data), np.max(all_data)
        set_range = (min_val, max_val)
    elif mode == "config_set_range":
        set_range = min_max

    ima_histograms = []
    nima_histograms = []

    # Create histograms for ima levels
    for ima_level in ima_levels:
        if ima_level is not None:
            hist, bin_edges = np.histogram(ima_level, bins=50, range=set_range, density=False)
            ima_histograms.append({
                'histogram': hist,
                'bin_edges': bin_edges,
                'mean': np.mean(ima_level),
                'min': np.min(ima_level),
                'max': np.max(ima_level)
            })
        else:
            ima_histograms.append(None)

    # Create histograms for nima levels
    for nima_level in nima_levels:
        if nima_level is not None:
            hist, bin_edges = np.histogram(nima_level, bins=50, range=set_range, density=False)
            nima_histograms.append({
                'histogram': hist,
                'bin_edges': bin_edges,
                'mean': np.mean(nima_level),
                'min': np.min(nima_level),
                'max': np.max(nima_level)
            })
        else:
            nima_histograms.append(None)

    # Create combined plots
    n_levels = len(ima_levels)
    if n_levels > 0:
        fig, axes = plt.subplots(2, (n_levels + 1) // 2, figsize=(15, 8))
        if n_levels == 1:
            axes = [axes]
        elif n_levels <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(n_levels):
            if i < len(axes):
                ax = axes[i]

                # Plot ima histogram
                if ima_histograms[i] is not None:
                    ima_hist = ima_histograms[i]
                    bin_centers = (ima_hist['bin_edges'][:-1] + ima_hist['bin_edges'][1:]) / 2
                    ax.bar(bin_centers, ima_hist['histogram'],
                          width=np.diff(ima_hist['bin_edges']),
                          alpha=0.6, edgecolor='blue', color='blue', label='IMA')

                # Plot nima histogram (overlaid)
                if nima_histograms[i] is not None:
                    nima_hist = nima_histograms[i]
                    bin_centers = (nima_hist['bin_edges'][:-1] + nima_hist['bin_edges'][1:]) / 2
                    ax.bar(bin_centers, nima_hist['histogram'],
                          width=np.diff(nima_hist['bin_edges']),
                          alpha=0.6, edgecolor='red', color='red', label='NIMA')

                # Add title with statistics
                title_text = f'{band_name.title()} - Level {i}\n'
                if ima_histograms[i] is not None:
                    title_text += f'IMA: Mean={ima_histograms[i]["mean"]:.0f}, Min={ima_histograms[i]["min"]:.1f}, Max={ima_histograms[i]["max"]:.1f}\n'
                if nima_histograms[i] is not None:
                    title_text += f'NIMA: Mean={nima_histograms[i]["mean"]:.0f}, Min={nima_histograms[i]["min"]:.1f}, Max={nima_histograms[i]["max"]:.1f}'

                ax.set_title(title_text)
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                ax.legend()

        # Hide extra subplots
        for i in range(n_levels, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(histograms_dir, f'{band_name}_ima_nima_combined_histograms.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def save_band_fourier_coeff_histograms_combined(
    band_name: str,
    statistics_dict: Dict,
    histograms_dir: str,
    direction: str = "vertical"  # "vertical" or "horizontal"
):
    """
    Save combined histograms and frequency plots for both original and damped Fourier coefficients.

    Args:
        band_name: Name of the band (e.g., 'red', 'green', 'blue')
        statistics_dict: Dictionary containing fourier coefficient data
        histograms_dir: Directory to save histogram plots
        direction: Direction for coefficients ("vertical" or "horizontal")
    """
    # Set up keys based on direction
    coeffs_key = f"{direction}_fourier_coeffs"
    coeffs_damped_key = f"{direction}_fourier_coeffs_damped"
    freqs_key = f"{direction}_fourier_frequencies"

    fourier_levels = statistics_dict[coeffs_key]
    fourier_levels_damped = statistics_dict[coeffs_damped_key]
    fourier_frequencies = statistics_dict[freqs_key]

    freq_magnitude_data_original = []
    freq_magnitude_data_damped = []

    # Process each decomposition level
    for i, (level_orig, level_damped, frequencies) in enumerate(
        zip(fourier_levels, fourier_levels_damped, fourier_frequencies)
    ):
        if level_orig is None or level_damped is None or frequencies is None:
            freq_magnitude_data_original.append(None)
            freq_magnitude_data_damped.append(None)
            continue

        # Process original coefficients
        magnitude_orig = np.abs(level_orig)
        if direction == "vertical":
            magnitude_profile_orig = np.mean(magnitude_orig, axis=1)
        else:
            magnitude_profile_orig = np.mean(magnitude_orig, axis=0)

        freq_magnitude_data_original.append({
            'frequencies': frequencies,
            'magnitude_profile': magnitude_profile_orig,
            'level': i
        })

        # Process damped coefficients
        magnitude_damped = np.abs(level_damped)
        if direction == "vertical":
            magnitude_profile_damped = np.mean(magnitude_damped, axis=1)
        else:
            magnitude_profile_damped = np.mean(magnitude_damped, axis=0)

        freq_magnitude_data_damped.append({
            'frequencies': frequencies,
            'magnitude_profile': magnitude_profile_damped,
            'level': i
        })

    # Plot combined frequency vs magnitude
    valid_orig_data = [d for d in freq_magnitude_data_original if d is not None]
    valid_damped_data = [d for d in freq_magnitude_data_damped if d is not None]
    n_levels = len(valid_orig_data)

    if n_levels > 0:
        cols = min(3, n_levels)
        rows = (n_levels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        if n_levels == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        for i, (orig_data, damped_data) in enumerate(zip(valid_orig_data, valid_damped_data)):
            if i < len(axes):
                ax = axes[i]
                level = orig_data['level']
                frequencies = orig_data['frequencies']

                # Plot original coefficients
                ax.plot(frequencies, orig_data['magnitude_profile'],
                       'b-', alpha=0.8, linewidth=2, label='Original')

                # Plot damped coefficients
                ax.plot(frequencies, damped_data['magnitude_profile'],
                       'r-', alpha=0.8, linewidth=2, label='Damped')

                ax.set_xlabel(f'{direction.title()} Frequency (cycles/pixel)')
                ax.set_ylabel('Magnitude')
                ax.set_title(f'{band_name.title()} - {direction.title()} FFT Level {level}\nOriginal vs Damped')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')

                # Highlight DC component
                ax.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='DC (0 Hz)')
                ax.legend()
                ax.set_xlim(frequencies.min(), frequencies.max())

        # Hide extra subplots
        for i in range(n_levels, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(histograms_dir, f'{band_name}_{direction}_fourier_original_vs_damped.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()


def save_histograms_and_debug_geotiffs_enhanced(all_statistics, alpha_statistics, profile, intermediates_dir, dec_num_v, dec_num_h):
    """
    Enhanced version that creates both individual and combined histogram plots.

    Args:
        all_statistics: List of statistics dictionaries from each RGB band
        alpha_statistics: Statistics dictionary from alpha band
        profile: Rasterio profile for GeoTIFF creation
        intermediates_dir: Directory to save files
        dec_num_v: Number of vertical decomposition levels
        dec_num_h: Number of horizontal decomposition levels
    """
    # Create directories for outputs
    histograms_dir = os.path.join(intermediates_dir, "histograms")
    debug_geotiffs_dir = os.path.join(intermediates_dir, "debug_geotiffs")
    os.makedirs(histograms_dir, exist_ok=True)
    os.makedirs(debug_geotiffs_dir, exist_ok=True)

    band_names = ['red', 'green', 'blue']
    max_dec_levels = max(dec_num_v, dec_num_h)

    # Create debug GeoTIFFs (unchanged)
    create_debug_geotiffs(all_statistics, alpha_statistics, profile, debug_geotiffs_dir, max_dec_levels)

    # Process RGB bands with enhanced histograms
    for band_idx in range(3):
        statistics = all_statistics[band_idx]
        band_name = band_names[band_idx]

        # Create combined ima/nima histograms
        save_band_image_histograms_combined(band_name, statistics, histograms_dir)

        # Create combined original/damped Fourier coefficient plots
        if dec_num_v > 0:
            save_band_fourier_coeff_histograms_combined(band_name, statistics, histograms_dir, "vertical")
        if dec_num_h > 0:
            save_band_fourier_coeff_histograms_combined(band_name, statistics, histograms_dir, "horizontal")

        # Still create individual plots for detailed analysis
        save_band_histograms(statistics, band_name, histograms_dir)

    save_band_image_histograms_combined(band_name, alpha_statistics, histograms_dir)
    save_band_fourier_coeff_histograms_combined(band_name, alpha_statistics, histograms_dir, "vertical")


# Additional utility function for comparing algorithm effectiveness
def create_algorithm_comparison_plots(all_statistics, histograms_dir):
    """
    Create summary plots showing the effectiveness of the destriping algorithm.

    Args:
        all_statistics: List of statistics dictionaries from each RGB band
        histograms_dir: Directory to save plots
    """
    band_names = ['red', 'green', 'blue']

    # Create a summary plot showing mean pixel values across decomposition levels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for band_idx, (statistics, band_name) in enumerate(zip(all_statistics, band_names)):
        ax = axes[band_idx]

        ima_means = []
        nima_means = []
        levels = []

        for i, (ima_level, nima_level) in enumerate(zip(statistics['ima_levels'], statistics['nima_levels'])):
            if ima_level is not None and nima_level is not None:
                ima_means.append(np.mean(ima_level))
                nima_means.append(np.mean(nima_level))
                levels.append(i)

        ax.plot(levels, ima_means, 'o-', label='IMA (Original)', color='blue')
        ax.plot(levels, nima_means, 's-', label='NIMA (Processed)', color='red')
        ax.set_xlabel('Decomposition Level')
        ax.set_ylabel('Mean Pixel Value')
        ax.set_title(f'{band_name.title()} Band - Mean Values')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(histograms_dir, 'algorithm_effectiveness_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    # output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_2/chlor_a_oceancolor_processed.tif"
    # intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_test_1/intermediates"
    # dec_num_v=5
    # dec_num_h=5
    # sigma_v=2.4
    # sigma_h=2.4
    # process_rgb(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)


    # geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    # output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    # intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_test_2/intermediates"
    # dec_num_v = 12
    # dec_num_h = 0
    # sigma_v = 12
    # sigma_h = 0
    # process_rgb(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)

    # geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    # output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    # intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_test_3/intermediates"
    # dec_num_v = 12
    # dec_num_h = 5
    # sigma_v = 2.4
    # sigma_h = 2.4
    # process_rgb(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)

    # geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    # output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    # intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_test_4/intermediates"
    # dec_num_v = 12
    # dec_num_h = 5
    # sigma_v = 12
    # sigma_h = 2.4
    # process_rgb(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)

    # geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    # output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    # intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_test_5/intermediates"
    # dec_num_v = 10
    # dec_num_h = 4
    # sigma_v = 12
    # sigma_h = 5
    # process_rgb(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)

    geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_2/chlor_a_oceancolor_processed.tif"
    intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_tests/modified_muench_test_1/intermediates"
    dec_num_v=6
    dec_num_h=0
    sigma_v=2.
    sigma_h=0.
    process_rgba(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)


    geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_tests/modified_muench_test_2/intermediates"
    dec_num_v = 9
    dec_num_h = 0
    sigma_v = 2.
    sigma_h = 0.
    process_rgba(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)

    geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_tests/modified_muench_test_3/intermediates"
    dec_num_v = 12
    dec_num_h = 0
    sigma_v = 2.
    sigma_h = 0.
    process_rgba(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)

    geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_tests/modified_muench_test_4/intermediates"
    dec_num_v = 6
    dec_num_h = 0
    sigma_v = 7.
    sigma_h = 0.
    process_rgba(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)

    geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_tests/modified_muench_test_5/intermediates"
    dec_num_v = 9
    dec_num_h = 0
    sigma_v = 7.
    sigma_h = 0
    process_rgba(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)

    geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_tests/modified_muench_test_6/intermediates"
    dec_num_v = 12
    dec_num_h = 0
    sigma_v = 7.
    sigma_h = 0
    process_rgba(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)

    geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_tests/modified_muench_test_7/intermediates"
    dec_num_v = 6
    dec_num_h = 0
    sigma_v = 12.
    sigma_h = 0.
    process_rgba(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)

    geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_tests/modified_muench_test_8/intermediates"
    dec_num_v = 9
    dec_num_h = 0
    sigma_v = 12.
    sigma_h = 0
    process_rgba(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)

    geotiff_path = "/Volumes/LaCie/test_destriper/test_data/chlor_a_oceancolor.tif"
    output_path = "/Volumes/LaCie/test_destriper/modified_muench_test_3/chlor_a_oceancolor_processed.tif"
    intermediates_dir = "/Volumes/LaCie/test_destriper/modified_muench_tests/modified_muench_test_9/intermediates"
    dec_num_v = 12
    dec_num_h = 0
    sigma_v = 12.
    sigma_h = 0
    process_rgba(geotiff_path, output_path, intermediates_dir, True, dec_num_v, dec_num_h, sigma_v, sigma_h)
