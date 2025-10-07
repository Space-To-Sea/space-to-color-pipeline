#!/bin/zsh

# Define the base directory and MATLAB script path
BASE_DIR="/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH"
MATLAB_GRAYSCALE_SCRIPT_PATH="/Users/alexfranks/Projects/UROP_Summer2024/NEW/dropbox_scripts/matlab_scripts/processing_singleimage.m"
MATLAB_RGB_SCRIPT_PATH="/Users/alexfranks/Projects/UROP_Summer2024/NEW/dropbox_scripts/matlab_scripts/processing_singleimage_auto.m"

# MATLAB executable (ensure it's in the PATH or use full path)
MATLAB_EXEC="matlab"

# List of files to move
FILES=(
    "seadas_products_RGB.tif"
    "seadas_products_diatoms_hirata.tif"
    "seadas_products_dinoflagellates_hirata.tif"
    "seadas_products_greenalgae_hirata.tif"
    "seadas_products_prymnesiophytes_hirata.tif"
    "seadas_products_chlor_a_oceancolor.tif"
)

# Iterate through each directory in the base directory
for dir in $BASE_DIR/Processing/*; do
    if [ -d "$dir/seadas" ]; then
        # Create the color directory if it doesn't exist
        mkdir -p "$dir/seadas/color"
        # Move the specified files to the color directory
        for file in "${FILES[@]}"; do
            if [ -f "$dir/seadas/$file" ]; then
                mv "$dir/seadas/$file" "$dir/seadas/color/"
                echo "Moved $file to $dir/seadas/color/"
            else
                echo "File $file does not exist in $dir/seadas"
            fi
        done

        # Define paths
        CURRENT_DIR="$dir/seadas"
        COLOR_DIR="$dir/seadas/color"
        CLOUD_FILE="$dir/seadas/seadas_products_cloud_albedo.tif"
        GRAY_CHLOR_A_FILE="$dir/seadas/seadas_products_chlor_a_gray_scale.tif"
        MATLAB_DIR="$dir/matlab"

        # Create the MATLAB directory if it doesn't exist
        mkdir -p "$MATLAB_DIR"

        # Grayscale processing
        sed -i "" "s|^current_path = .*|current_path = '$CURRENT_DIR';|" "$MATLAB_GRAYSCALE_SCRIPT_PATH"
        sed -i "" "s|^savefolder = .*|savefolder = '$MATLAB_DIR';|" "$MATLAB_GRAYSCALE_SCRIPT_PATH"

        # Cloud albedo paths
        sed -i "" "s|^original_image_name = .*|original_image_name = '$CLOUD_FILE';|" "$MATLAB_GRAYSCALE_SCRIPT_PATH"
        sed -i "" "s|^savefilename = .*|savefilename = 'cloud_albedo_stripe_corrected.tif';|" "$MATLAB_GRAYSCALE_SCRIPT_PATH"

        # Run the MATLAB script for cloud albedo
        "$MATLAB_EXEC" -nodisplay -nosplash -r "run('$MATLAB_GRAYSCALE_SCRIPT_PATH'); exit;"

        # Chlor a paths
        sed -i "" "s|^original_image_name = .*|original_image_name = '$GRAY_CHLOR_A_FILE';|" "$MATLAB_GRAYSCALE_SCRIPT_PATH"
        sed -i "" "s|^savefilename = .*|savefilename = 'chlor_a_gray_scale_stripe_corrected.tif';|" "$MATLAB_GRAYSCALE_SCRIPT_PATH"

        # Run the MATLAB script for chlor a
        "$MATLAB_EXEC" -nodisplay -nosplash -r "run('$MATLAB_GRAYSCALE_SCRIPT_PATH'); exit;"

        # Color processing
        sed -i "" "s|^current_path = .*|current_path = '$COLOR_DIR';|" "$MATLAB_RGB_SCRIPT_PATH"
        sed -i "" "s|^original_images_names = .*|original_images_names = {'$COLOR_DIR/seadas_products_RGB.tif', '$COLOR_DIR/seadas_products_chlor_a_oceancolor.tif', '$COLOR_DIR/seadas_products_diatoms_hirata.tif', '$COLOR_DIR/seadas_products_dinoflagellates_hirata.tif', '$COLOR_DIR/seadas_products_greenalgae_hirata.tif', '$COLOR_DIR/seadas_products_prymnesiophytes_hirata.tif'};|" "$MATLAB_RGB_SCRIPT_PATH"
        sed -i "" "s|^savefolder = .*|savefolder = '$MATLAB_DIR';|" "$MATLAB_RGB_SCRIPT_PATH"
        sed -i "" "s|^savefilenames = .*|savefilenames = {'RGB_stripe_corrected.tif','chlor_a_oceancolor_stripe_corrected.tif','diatoms_stripe_corrected.tif','dinoflagellates_stripe_corrected.tif','greenalgae_stripe_corrected.tif','prymnesiophytes_stripe_corrected.tif'};|" "$MATLAB_RGB_SCRIPT_PATH"

        # Run the MATLAB script
        "$MATLAB_EXEC" -nodisplay -nosplash -r "run('$MATLAB_RGB_SCRIPT_PATH'); exit;"
    else
        echo "Directory $dir/seadas does not exist"
    fi
done
