```python
#!/usr/bin/env python3
"""
run_experiment.py

run one muench destriping experiment with specific parameters.

use this when you want to test one exact setting instead of running a sweep.

Example:
    python run_experiment.py \
        --input test_data/chlor_a_oceancolor.tif \
        --out-dir experiments/manual_test \
        --pipeline-module modified_muench \
        --dec-v 6 \
        --sigma-v 2.4 \
        --dec-h 0 \
        --sigma-h 0 \
        --enhance-contrast
"""

import argparse
import json
from pathlib import Path

from experiment_utils import clean_float, safe_import_process_rgba


def main():
    # set up the command line arguments
    parser = argparse.ArgumentParser(description="Run one Muench destriping experiment.")

    # input image and where we want the results to go
    parser.add_argument("--input", required=True, help="Input RGBA GeoTIFF path")
    parser.add_argument("--out-dir", required=True, help="Output folder for this one experiment")

    # choose which pipeline file has process_rgba()
    parser.add_argument(
        "--pipeline-module",
        default="modified_muench",
        help="Python module that contains process_rgba(), without .py"
    )

    # exact vertical and horizontal destriping settings
    parser.add_argument("--dec-v", type=int, required=True)
    parser.add_argument("--sigma-v", type=float, required=True)
    parser.add_argument("--dec-h", type=int, default=0)
    parser.add_argument("--sigma-h", type=float, default=0.0)

    # optional extras
    parser.add_argument("--enhance-contrast", action="store_true")
    parser.add_argument("--no-statistics", action="store_true")

    args = parser.parse_args()

    # load process_rgba() from the selected pipeline
    process_rgba = safe_import_process_rgba(args.pipeline_module)

    # clean up the paths before using them
    input_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()

    # make the output folder if needed
    out_dir.mkdir(parents=True, exist_ok=True)

    # make sure the input image is actually there
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # if horizontal destriping is off, sigma_h doesn't really matter
    sigma_h = 0.0 if args.dec_h == 0 else args.sigma_h

    # put the main parameter values into the run name
    run_name = (
        f"manual_"
        f"dv{args.dec_v:02d}_"
        f"sv{clean_float(args.sigma_v)}_"
        f"dh{args.dec_h:02d}_"
        f"sh{clean_float(sigma_h)}"
    )

    # set up output and intermediate file locations
    intermediates_dir = out_dir / "intermediates"
    output_path = out_dir / f"{input_path.stem}_{run_name}_processed.tif"

    # make the intermediate folder if needed
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    # save all the settings used for this run
    params = {
        "run_name": run_name,
        "dec_num_v": args.dec_v,
        "sigma_v": args.sigma_v,
        "dec_num_h": args.dec_h,
        "sigma_h": sigma_h,
        "enhance_contrast": args.enhance_contrast,
        "save_statistics": not args.no_statistics,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "intermediates_dir": str(intermediates_dir),
    }

    # write the settings out so we know exactly what was used later
    with open(out_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"Running one experiment: {run_name}")
    print(f"Output: {output_path}")

    # run the actual destriping pipeline
    process_rgba(
        input_path=str(input_path),
        output_path=str(output_path),
        intermediates_dir=str(intermediates_dir),
        enhance_contrast=args.enhance_contrast,
        dec_num_v=args.dec_v,
        dec_num_h=args.dec_h,
        sigma_v=args.sigma_v,
        sigma_h=sigma_h,
        save_statistics=not args.no_statistics,
    )

    print("Finished.")


if __name__ == "__main__":
    main()
```
