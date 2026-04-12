import dropbox
import os
from dropbox.files import FileMetadata, FolderMetadata
from dropbox import DropboxOAuth2FlowNoRedirect


def download_file(dbx,dbx_path, local_path):
    """Download a single file from Dropbox in chunks (good for large files)."""
    print(f"Downloading: {dbx_path} -> {local_path}")
    try:
        metadata, res = dbx.files_download(dbx_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True) #makes the download folder if it doesn't exist
        
        #download the file in chunks (good for large files)
        with open(local_path, 'wb') as f:
            for chunk in res.iter_content(CHUNK_SIZE):
                f.write(chunk)
    except Exception as e:
        print(f"Error downloading {dbx_path}: {e}")
        
def download_list(dbx,result,local_folder):
    for entry in result.entries: 
        local_path = os.path.join(local_folder, entry.name)
        
        if isinstance(entry, FileMetadata):
            if os.path.exists(local_path) and os.path.getsize(local_path) == entry.size:
                print(f"Skipping (already exists): {local_path}")
                continue
            download_file(dbx,entry.path_lower, local_path) #if it is a file, download it
        elif isinstance(entry, FolderMetadata):
            download_folder(dbx,entry.path_lower, local_path) #if it is a subfolder, download the files inside that subfolder

def download_folder(dbx,dbx_folder, local_folder):
    """Download all files in a Dropbox folder."""
    try:
        result = dbx.files_list_folder(dbx_folder) #collect a list of all files in the folder
    except dropbox.exceptions.ApiError as e:
        print(f"Failed to list folder {dbx_folder}: {e}")
        return
    
    download_list(dbx,result,local_folder)

    #result=dbx.files_lsit_folder(dbx_folder) only returns the first 2000 elements in the folder.
    #if there are more than 2000 files, this next section iterates through them
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        
        download_list(dbx,result,local_folder)
        
def first_time_config(APP_KEY, APP_SECRET):
    flow = DropboxOAuth2FlowNoRedirect(
        APP_KEY,
        APP_SECRET,
        token_access_type="offline"
    )

    # Step A: open this URL in your browser
    authorize_url = flow.start()
    print("1) Go to this URL and click Allow:")
    print(authorize_url)

    # Step B: Dropbox shows you a short code
    auth_code = input("2) Paste the authorization code here: ").strip()

    # Step C: exchange code for tokens
    oauth_result = flow.finish(auth_code)

    print("\nSAVE THIS — YOU WILL NOT SEE IT AGAIN")
    print("Refresh token:", oauth_result.refresh_token)

if __name__ == '__main__':
    # === CONFIG ===
    #If this is your first time running this script, you need to set your APP_KEY, APP_SECRET, and oauth2_refresh_token
    #To find your APP_KEY and APP_SECRET, follow the instructions below:
    #Never share your app_key and app_secret with anyone else. Remove it before you upload this script anywhere
        #Go to https://www.dropbox.com/developers
        #Create an app with Full Dropbox access
        #Copy the app key and app secret below
    APP_KEY = "*****"
    APP_SECRET = "*****"
    
    # to find your oauth2_refresh_token, uncomment the two lines of code below and run them. Follow the instructions to get your refresh token.
    #first_time_config(APP_KEY,APP_SECRET)
    ##%%
    
    #Once you get your refresh token, copy it below. Then comment out the two lines above again.
    REFRESH_TOKEN="*****"
    
    dbx = dropbox.Dropbox(
    app_key=APP_KEY,
    app_secret=APP_SECRET,
    oauth2_refresh_token=REFRESH_TOKEN
)

    # === SET PATHS ===
    #You can customize DROPBOX_FOLDER (dropbox folder you download from) and DOWNLOAD_FOLDER (folder you download to)

    #change this to exact path to the specific Dropbox folder you want to download
    DROPBOX_FOLDER = r"/Space To Sea Summer 2021/Data Visualization Jan 2025/real_csvs"  # path inside the shared folder

    #change this to the folder you want to download things to. 
    DOWNLOAD_FOLDER = "F:/UROP/download_test"
    #to download to an external hard drive:
        #On Windows, external hard drives are usually assigned a drive letter, like E:\ or F:\.
        #Sample Windows path: DOWNLOAD_FOLDER="F:/UROP/download_test"
        #On Mac, external drives are usually mounted under /Volumes/
        #Sample Mac path: DOWNLOAD_FOLDER="/Volumes/ExternalDriveName/UROP/download_test"

    CHUNK_SIZE = 4 * 1024 * 1024     # Chunk size (4 MB recommended for large files)
    
    download_folder(dbx,DROPBOX_FOLDER, DOWNLOAD_FOLDER)