import dropbox
import os
from dropbox.files import WriteMode
from dropbox.common import PathRoot


# Taken fron download_from_drop.py
def first_time_config(APP_KEY, APP_SECRET):
    flow = dropbox.DropboxOAuth2FlowNoRedirect(
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
    

# ================= 

# SHARED_FOLDER_ID = "9544754880" # ID for folder Space to Sea Summer 2021
DROPBOX_SUBFOLDER_PATH = r"/Space To Sea Summer 2021/Image Processing/Destriping_Tests/Parameter_Variation_Testing_Muench"  # path inside the shared folder

# Chunk size (4 MB recommended for large files)
CHUNK_SIZE = 4 * 1024 * 1024

# =================

def upload_file(dbx, file_path, folder_name):
    DROPBOX_FILE_NAME = os.path.basename(file_path)

    file_size = os.path.getsize(file_path)
    print(f"Uploading '{file_path}' ({file_size / 1024 / 1024:.2f} MB) ...")

    with open(file_path, "rb") as f:
        if file_size <= CHUNK_SIZE:
            # Small file fallback
            dropbox_path = (
                            f"{DROPBOX_SUBFOLDER_PATH}/"
                            f"{folder_name}/"
                            f"{DROPBOX_FILE_NAME}"
                            )
            dbx.files_upload(f.read(), dropbox_path, mode=WriteMode.add) #note, it uploads the file only if it is unique (doesn't exist already)
        else:
            # Start upload session
            session = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
            cursor = dropbox.files.UploadSessionCursor(session.session_id, f.tell())
            commit = dropbox.files.CommitInfo(path=f"{DROPBOX_SUBFOLDER_PATH}/{folder_name}/{DROPBOX_FILE_NAME}", mode=WriteMode.add)
            
            while f.tell() < file_size:
                progress = f.tell() / file_size * 100
                print(f"Uploaded {progress:.2f}%")
                if (file_size - f.tell()) <= CHUNK_SIZE:
                    dbx.files_upload_session_finish(f.read(CHUNK_SIZE), cursor, commit)
                else:
                    dbx.files_upload_session_append_v2(f.read(CHUNK_SIZE), cursor)
                    cursor.offset = f.tell()

    print("Upload complete!")
  


def get_dbx_folders(dropbox_path):
    folders = []

    result = dbx.files_list_folder(dropbox_path)

    while True:
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FolderMetadata):
                folders.append(entry.name)
        if not result.has_more:
            break

    result = dbx.files_list_folder_continue(result.cursor)

    return folders

if __name__ == "__main__":

    #NOTE
    #To find your APP_KEY and APP_SECRET, follow the instructions below:
    #Never share your app_key and app_secret with anyone else. Remove it before you upload this script anywhere
        #Go to https://www.dropbox.com/developers
        #Create an app with Full Dropbox access
        #Copy the app key and app secret below
    APP_KEY = "***"
    APP_SECRET = "***"


    #NOTE
    #If first time running scrip uncomment the line below and follow the instructions
    #first_time_config(APP_KEY,APP_SECRET)
    

    #Once you get your refresh token, copy it below. Then comment out the two lines above again.
    REFRESH_TOKEN="***"

    dbx = dropbox.Dropbox(
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        oauth2_refresh_token=REFRESH_TOKEN
    )


    #NOTE
    # Make this the path to the folder containing all the folders with test images 
    # e.g. /USER/A/B/C/Test_Images/
    # this folder should contain the other folders like test01_a_b_c
    

    FOLDER_PATH = r"/Users/jmrivera/Downloads/temp" 


    already_uploaded = get_dbx_folders(DROPBOX_SUBFOLDER_PATH)

    for folder in os.listdir(FOLDER_PATH):
        if folder not in already_uploaded:           
            temp_folder_path = os.path.join(FOLDER_PATH,folder)
            for file in os.listdir(temp_folder_path):
                if file.endswith(".jpg"):    # you can this to upload other types of files too         
                    file_path = os.path.join(temp_folder_path,file)
                    upload_file(dbx, file_path, folder)
        else:
            print(f"Skipping {folder} (already uploaded)")


    
