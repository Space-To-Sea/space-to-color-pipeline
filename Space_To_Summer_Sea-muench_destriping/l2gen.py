import os
import subprocess
import numpy as np
from netCDF4 import Dataset

params = {
    "batch_path":"/Volumes/LaCie/Processing/0910_L2GEN_TEST_BATCH" #Add the Path to your Batch folder (Must contain a folder called Processing that contains the folders for each of your dates)
}
print(params["batch_path"])

def gdal_translate(input_file, output_file):
    """
    runs gdal_translate from terminal for an input and output file
    """
    command = [
        'gdal_translate',
        '-of', 'NetCDF',  # Specify output format as NetCDF
        input_file,
        output_file
    ]
    subprocess.run(command, check=True)

def l2gen(par_path):
    """
    runs l2gen from terminal with a generated par file
    """
    command = ['l2gen', f'par={par_path}']
    subprocess.run(command, check=True)

def watermask_tif_to_nc(preseadas_path_param, seadas_path_param):
    """
    converts the usgs provided watermask tif file in preseadas folder to a netcdf in seadas folder
    """
    nc_path = os.path.join(seadas_path_param, "WATER_MASK.nc")
    for file in os.listdir(preseadas_path_param):
        if file.startswith("._"):
            os.remove(os.path.join(preseadas_path_param, file))
        elif file.endswith("MTL.txt"):
            MTL_file_path = os.path.join(preseadas_path_param, file)
        elif file.endswith("WATER_MASK.tif") and not os.path.exists(nc_path):
            gdal_translate(os.path.join(preseadas_path_param, file), os.path.join(seadas_path_param, "WATER_MASK.nc"))
    return nc_path, MTL_file_path

def add_masks_to_nc(input_nc):
    """
    converts watermask.tif Band1 variable array to land, water, and cloud (unused) masks after tif->netcdf conversion which
    allows the netcdf to be taken as input in the l2gen par file
    """
    # Open the NetCDF file in append mode
    with Dataset(input_nc, 'a') as nc:
        if 'watermask' in nc.variables:
            return 0
        # Read the Band1 variable
        band1 = nc.variables['Band1'][:]
        # Check if the new variables already exist, if so delete them
        if 'watermask' in nc.variables:
            del nc.variables['watermask']
        if 'landmask' in nc.variables:
            del nc.variables['landmask']
        # Create the watermask variable
        watermask = nc.createVariable('watermask', 'b', ('y', 'x'), fill_value=-1)
        watermask.long_name = "watermask"
        watermask.description = "A simple binary water mask"
        watermask.comment = "0 = land, 1 = water"
        watermask.valid_min = 0
        watermask.valid_max = 1
        # Create the landmask variable
        landmask = nc.createVariable('landmask', 'b', ('y', 'x'), fill_value=-1)
        landmask.long_name = "landmask"
        landmask.description = "A simple binary land mask"
        landmask.comment = "0 = water, 1 = land"
        landmask.valid_min = 0
        landmask.valid_max = 1
        # Create the cloudmask variable
        cloudmask = nc.createVariable('cloudmask', 'b', ('y', 'x'), fill_value=-1)
        cloudmask.long_name = "cloudmask"
        cloudmask.description = "A simple binary cloud mask"
        cloudmask.comment = "0 = not clouds, 1 = clouds"
        cloudmask.valid_min = 0
        cloudmask.valid_max = 1
        # Create the shadowmask variable
        shadowmask = nc.createVariable('shadowmask', 'b', ('y', 'x'), fill_value=-1)
        shadowmask.long_name = "shadowmask"
        shadowmask.description = "A simple binary shadow mask"
        shadowmask.comment = "0 = not shadow, 1 = shadow"
        shadowmask.valid_min = 0
        shadowmask.valid_max = 1
        # Fill the watermask and landmask variables
        watermask_data = np.where(band1 == 1, 1, 0).astype('b')
        landmask_data = np.where(band1 == 0, 1, 0).astype('b')
        cloudmask_data = np.where(band1 == 2, 1, 0).astype('b')
        shadowmask_data = np.where(band1 == 3, 1, 0).astype('b')
        # Assign the data to the variables
        watermask[:] = watermask_data
        landmask[:] = landmask_data
        cloudmask[:] = cloudmask_data
        shadowmask[:] = shadowmask_data

def make_masks(preseadas_path_param, seadas_path_param):
    """
    makes a l2gen usable mask for a particular date given preseadas and seadas paths
    """
    nc_path, mtl_path = watermask_tif_to_nc(preseadas_path_param, seadas_path_param)
    add_masks_to_nc(nc_path)
    return nc_path, mtl_path

def run_batch_l2gen(batch_processing_path):
    """
    iterates over the folders in the Processing directory inside the batch_processing_path
    for each folder
        makes the l2gen usable mask
        makes the par file
        runs l2gen
    """
    folders = os.listdir(batch_processing_path)
    date_folders = [folder for folder in folders if folder[0] != "."]
    for folder in date_folders:
        folder_path = os.path.join(batch_processing_path, folder)
        print(f'checking {folder_path}')
        seadas_products_path = os.path.join(folder_path, "seadas/seadas_products.nc")
        preseadas_directory_path = os.path.join(folder_path, "preseadas")
        seadas_directory_path = os.path.join(folder_path, "seadas")
        mask_path, mtl_path = make_masks(preseadas_directory_path, seadas_directory_path)

        if os.path.isdir(seadas_directory_path) and not os.path.isfile(seadas_products_path):
            print(f'running l2gen for {folder_path}')

            ifile = mtl_path

            if ifile:
                ofile = os.path.join(seadas_directory_path, "seadas_products.nc")
                water = land = mask_path

                # Create the content for the .par file
                content = f"""# PRIMARY INPUT OUTPUT FIELDS
ifile={ifile}
ofile={ofile}

# SUITE
suite=OC

# PRODUCTS
l2prod=chlor_a cloud_albedo diatoms_hirata dinoflagellates_hirata greenalgae_hirata prymnesiophytes_hirata rhos_nnn

# ANCILLARY INPUTS  Default = climatology (select 'Get Ancillary' to download ancillary files)
land={land}
water={water}
"""

                # Define the output .par file path
                par_file_path = os.path.join(seadas_directory_path, "config.par")

                # Write the content to the .par file
                with open(par_file_path, 'w') as par_file:
                    par_file.write(content)

                print(f".par file generated at {par_file_path}")

                # Run l2gen with the generated .par file
                l2gen(par_file_path)

# Run the function
run_batch_l2gen(os.path.join(params["batch_path"], "Processing"))
