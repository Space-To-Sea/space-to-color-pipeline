#!/usr/bin/env python

import os
import sys

import yaml
from pathlib import Path
from typing import Dict, List


from flag import flag
from nc_to_gtiff import nc_to_gtiff
from interpolation import interpolation
# from palette_map import palette_map
# from palette_map_interpolated import palette_map_interpolated
from rotation import rotate_gtiffs
from masking import create_land_clouds_mask
from csv_export import csv_export
# from get_imas import ima_extraction
from stripe_correction import stripe_correction_land_cloud_reqs, stripe_correction_hirata, stripe_correction_chlor_a#, stripe_correction_setup
# from palette_map_destriped import palette_map_destriped
from palette_map import palette_map


def check_setup(config: Dict) -> None:
    if not (sys.version_info[:3] == (3, 12, 7)):
        print(f"Python 3.12.7 required, but running {sys.version_info}")
    elif config['verbose']['setup']:
        print(f"Python 3.12.7 requirement met")
    else:
        pass

    batch_dir_path = config['paths']['batch_dir']
    dates_dir_path = os.path.join(config['paths']['batch_dir'], config['paths']['dates_dir_rel_path'])

    assert os.path.exists(batch_dir_path)
    assert os.path.exists(dates_dir_path)


def get_date_dir_paths(config: Dict) -> List[str]:
    date_dir_paths = [
        os.path.join(os.path.join(config['paths']['batch_dir'], config['paths']['dates_dir_rel_path']), name)
        for name in os.listdir(os.path.join(config['paths']['batch_dir'], config['paths']['dates_dir_rel_path']))
        if os.path.isdir(os.path.join(os.path.join(config['paths']['batch_dir'], config['paths']['dates_dir_rel_path']), name))
    ]
    return date_dir_paths


def main():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    rerun_from_step = config['rerun_from_step']

    check_setup(config)
    date_dir_paths = get_date_dir_paths(config)
    for date_dir_path in date_dir_paths:
        print(f"\nProcessing date directory: {os.path.basename(date_dir_path)}")
        if rerun_from_step == 1:
            print("Step 1: Converting NetCDF to GeoTIFF...")
            nc_to_gtiff(config, date_dir_path)
        if rerun_from_step <= 2:
            print("Step 2: Exporting csvs...")
            csv_export(config, date_dir_path)
        if rerun_from_step <= 3:
            print("Step 3: Processing flags...")
            flag(config, date_dir_path)
        if rerun_from_step <= 4:
            print("Step 4: Rotating GeoTIFFs...")
            rotate_gtiffs(config, date_dir_path)
        if rerun_from_step <= 5:
            print("Step 5: ")
            # stripe_correction_setup(config, date_dir_path)
            stripe_correction_land_cloud_reqs(config, date_dir_path)
        if rerun_from_step <= 6:
            print("Step 6: Making masks...")
            create_land_clouds_mask(config, date_dir_path)
        if rerun_from_step <= 7:
            print("Step 7: Interpolating missing values...")
            # ima_extraction(config, date_dir_path)
            interpolation(config, date_dir_path)
        if rerun_from_step <= 8:
            print("Step 8:")
            # palette_map(config, date_dir_path)
            # palette_map_interpolated(config, date_dir_path)
            stripe_correction_chlor_a(config, date_dir_path)
            stripe_correction_hirata(config, date_dir_path)
            # palette_map_destriped(config, date_dir_path)
        if rerun_from_step <= 9:
            print("Step 9:")
            palette_map(config, date_dir_path)

if __name__ == "__main__":
    main()
