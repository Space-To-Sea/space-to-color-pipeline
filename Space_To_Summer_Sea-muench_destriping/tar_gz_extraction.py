import os
import sys
import tarfile

params = {
    "gz_files_directory_path": "/Volumes/LaCie/Processing/0910_TARS_TEST_BATCH/tars", #REPLACE PATH/TO/GZ/DIRECTORY with the correct absolute PATH (in reference to ~ or /) to the folder containing the tar.gz files like /Volumes/LaCie/Processing/BATCH_08_25_2024/tars/
}

def extract_tar_gz_files(tar_gz_files_param, output_directory_param): #output_directory_param should be the return of create_output_directory and is a path string
    for tar_gz_file in tar_gz_files_param:
        file_name = os.path.basename(tar_gz_file)
        folder_name = f"{file_name[14:16]}_{file_name[16:18]}_{file_name[10:14]}"
        date_folder = os.path.join(output_directory_param, folder_name)
        os.makedirs(date_folder)

        #makesubfolder
        extract_folder = os.path.join(date_folder, "preseadas")
        os.makedirs(extract_folder)
        os.makedirs(os.path.join(date_folder, "seadas"))
        os.makedirs(os.path.join(date_folder, "matlab"))
        os.makedirs(os.path.join(date_folder, "photoshop"))
        final_folder = os.path.join(date_folder, "final")
        os.makedirs(final_folder)
        os.makedirs(os.path.join(final_folder, folder_name))

        with tarfile.open(tar_gz_file, "r:gz") as tar:
            tar.extractall(path=extract_folder)
            print(f"Extracted {tar_gz_file} to {extract_folder}")

        # Delete the original .tar.gz file after extraction
        os.remove(tar_gz_file)
        print(f"Deleted original file: {tar_gz_file}")

def main():
    gz_files_directory = params["gz_files_directory_path"]
    tar_gz_files = [os.path.join(gz_files_directory, file) for file in os.listdir(gz_files_directory) if file[0] != "."]
    batch_path = os.path.dirname(params["gz_files_directory_path"])
    print(f"{params["gz_files_directory_path"]=}")
    print(f"{os.path.dirname(params["gz_files_directory_path"])=}")
    output_directory = os.path.join(os.path.dirname(params["gz_files_directory_path"]), "Processing")
    #Error checking
    print({file:file.endswith('.tar.gz') for file in tar_gz_files})
    if not all([file.endswith('.tar.gz') for file in tar_gz_files]):
        print("An unknown file extension is in the this directory")
        sys.exit()
    if not tar_gz_files:
        print("No .tar.gz files found in tar directory")
        sys.exit()
    print(f"{os.path.basename(batch_path)=}")
    if "BATCH" not in os.path.basename(batch_path):
        print("Parent title doesn't include BATCH")
        sys.exit()
    os.makedirs(output_directory)
    extract_tar_gz_files(tar_gz_files, output_directory)
    print(f"All .tar.gz files have been extracted to {output_directory}")

if __name__ == "__main__":
    main()
