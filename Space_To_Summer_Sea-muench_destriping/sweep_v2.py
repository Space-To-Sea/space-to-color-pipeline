#!/usr/bin/env python3

import argparse
import csv
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import rasterio

from experiment_utils import latin_hypercube_samples, clean_float, safe_import_process_rgba


def get_next_test_number(out_root: Path) -> int:
    # check existing test folders so nothing gets overwritten
    existing = []
    pattern = re.compile(r"test_(\d{3})")

    for folder in out_root.glob("test_*"):
        if folder.is_dir():
            match = pattern.match(folder.name)
            if match:
                existing.append(int(match.group(1)))

    return max(existing, default=0) + 1


def main():
    parser = argparse.ArgumentParser(
        description="Run Muench destriping tests and append new test folders."
    )

    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--pipeline-module", default="modified_muench")
    parser.add_argument("--wavelet", default="db15")
    parser.add_argument("--rotation-angle", type=float, default=14.3)

    parser.add_argument("--dec-v-min", type=int, default=4)
    parser.add_argument("--dec-v-max", type=int, default=12)
    parser.add_argument("--sigma-v-min", type=float, default=0.5)
    parser.add_argument("--sigma-v-max", type=float, default=15.0)

    parser.add_argument("--enhance-contrast", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    process_rgba = safe_import_process_rgba(args.pipeline_module)

    input_folder = Path(args.input_folder).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_folder.glob("*.tif"))

    if not image_paths:
        raise FileNotFoundError(f"No .tif files found in {input_folder}")

    start_number = get_next_test_number(out_root)

    dec_v_values = latin_hypercube_samples(
        args.runs,
        args.dec_v_min,
        args.dec_v_max,
        is_int=True
    )

    sigma_v_values = latin_hypercube_samples(
        args.runs,
        args.sigma_v_min,
        args.sigma_v_max,
        is_int=False
    )

    csv_path = out_root / "parameter_tests.csv"
    csv_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not csv_exists:
            writer.writerow([
                "test_number",
                "folder_name",
                "wavelet",
                "L_dec_num_v",
                "sigma_v",
                "dec_num_h",
                "sigma_h"
            ])

        for run_idx in range(args.runs):
            test_num_int = start_number + run_idx
            test_number = f"{test_num_int:03d}"

            dec_num_v = dec_v_values[run_idx]
            sigma_v = round(sigma_v_values[run_idx], 2)

            dec_num_h = 0
            sigma_h = 0.0
            wavelet = args.wavelet

            run_name = (
                f"test_{test_number}_"
                f"L{dec_num_v}_"
                f"s{clean_float(sigma_v)}_"
                f"{wavelet}"
            )

            run_dir = out_root / run_name
            run_dir.mkdir(parents=True, exist_ok=False)

            writer.writerow([
                test_number,
                run_name,
                wavelet,
                dec_num_v,
                sigma_v,
                dec_num_h,
                sigma_h
            ])
            csvfile.flush()

            print("\n====================================")
            print(f"Running {run_name}")
            print("====================================")
            print(f"Wavelet: {wavelet}")
            print(f"L:       {dec_num_v}")
            print(f"Sigma:   {sigma_v}")
            print("Horizontal destriping: OFF")

            for image_path in image_paths:
                print(f"Processing {image_path.name}")

                temp_intermediates = run_dir / "_temp_intermediates"
                temp_output_tif = run_dir / f"_temp_{image_path.stem}.tif"
                output_jpg = run_dir / f"{image_path.stem}_processed.jpg"

                process_rgba(
                    input_path=str(image_path),
                    output_path=str(temp_output_tif),
                    intermediates_dir=str(temp_intermediates),
                    enhance_contrast=args.enhance_contrast,
                    dec_num_v=dec_num_v,
                    dec_num_h=dec_num_h,
                    sigma_v=sigma_v,
                    sigma_h=sigma_h,
                    save_statistics=False,
                    wavelet=wavelet,
                    rotation_angle=args.rotation_angle,
                )

                with rasterio.open(temp_output_tif) as src:
                    data = src.read()

                rgb = np.transpose(data[:3], (1, 2, 0))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                cv2.imwrite(
                    str(output_jpg),
                    bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )

                if temp_output_tif.exists():
                    temp_output_tif.unlink()

                if temp_intermediates.exists():
                    shutil.rmtree(temp_intermediates)

    print("\nDone.")
    print(f"CSV saved/appended at: {csv_path}")


if __name__ == "__main__":
    main()
