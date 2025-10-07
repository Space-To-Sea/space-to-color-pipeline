import os
import shutil
import glob

import numpy as np
import rasterio
from typing import Dict

def save_bit_flags(bit_num_map: Dict, input_gtiff_path: str, output_dir_path: str, verbose: bool) -> None:
    with rasterio.open(input_gtiff_path) as src:
        flags_data = src.read(1)
        profile = src.profile.copy()

        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=255
        )

        max_value = np.max(flags_data)
        num_bits = int(np.ceil(np.log2(max_value + 1)))

        num_bits = max(num_bits, 1)

        for bit_pos in range(num_bits):

            bit_mask = ((flags_data >> bit_pos) & 1).astype(np.uint8)

            output_path = os.path.join(output_dir_path, f"{bit_num_map[bit_pos]}.tif")

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(bit_mask, 1)

def flag(config: Dict, date_dir_path: str):
    flags_out_dir_path = os.path.join(date_dir_path, config['paths']['flags_out_rel_path'])

    if os.path.exists(flags_out_dir_path):
        shutil.rmtree(flags_out_dir_path, ignore_errors=True)

    os.makedirs(flags_out_dir_path)

    flag_config = config['flags']


    # handle preseadas derived

    preseadas_bit_num_map = flag_config['flag_levels'][flag_config['preseadas_derived']['flag_level']]
    preseadas_input_gtiff_path = glob.glob(os.path.join(date_dir_path, config['paths']['l2flags_tif_path_search']))[0]
    preseadas_output_dir_path = os.path.join(flags_out_dir_path, flag_config['preseadas_derived']['save_folder_rel_path'])
    verbose = config['verbose']['flag']

    os.makedirs(preseadas_output_dir_path)

    save_bit_flags(preseadas_bit_num_map, preseadas_input_gtiff_path, preseadas_output_dir_path, verbose)

    # handle seadas products nc derived

    seadas_products_nc_bit_num_map = flag_config['flag_levels'][flag_config['seadas_products_nc_derived']['flag_level']]
    seadas_products_nc_input_gtiff_path = os.path.join(date_dir_path, config['paths']['nc_to_gtiff_out_rel_path'], config['nc_to_gtiff']['nc_exports']['l2_flags']['save_name'])
    seadas_products_nc_output_dir_path = os.path.join(flags_out_dir_path, flag_config['seadas_products_nc_derived']['save_folder_rel_path'])
    verbose = config['verbose']['flag']

    os.makedirs(seadas_products_nc_output_dir_path)

    save_bit_flags(seadas_products_nc_bit_num_map, seadas_products_nc_input_gtiff_path, seadas_products_nc_output_dir_path, verbose)

    mask_out_dir_path = os.path.join(flags_out_dir_path, flag_config['masks_out_rel_path'])
    os.makedirs(mask_out_dir_path)

    for mask_name, mask_path in flag_config['masks'].items():
        mask_input_path = os.path.join(flags_out_dir_path, mask_path)
        mask_output_path = os.path.join(mask_out_dir_path, f"{mask_name}.tif")
        # os.symlink(mask_input_path, mask_output_path)
        shutil.copy2(mask_input_path, mask_output_path)
