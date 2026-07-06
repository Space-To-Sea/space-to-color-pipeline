import dropbox
import os
from dropbox.files import WriteMode
from dropbox.common import PathRoot

# === CONFIG ===
# Each user should paste their own access token as a string here
#Never share your access_token with anyone else. Remove it before you upload this script anywhere
#Find your access token by:
    #going to https://www.dropbox.com/developers
    #Create an app with Full Dropbox access
    #Generate and copy the access token below
ACCESS_TOKEN="*****"

dbx = dropbox.Dropbox(ACCESS_TOKEN)

# SHARED_FOLDER_ID = "9544754880" # ID for folder Space to Sea Summer 2021
DROPBOX_SUBFOLDER_PATH = r"/Space To Sea Summer 2021/Data Visualization Jan 2025/real_csvs"  # path inside the shared folder

# Chunk size (4 MB recommended for large files)
CHUNK_SIZE = 4 * 1024 * 1024

# =================

def upload_file(file_path):
    # Connect to Dropbox
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    DROPBOX_FILE_NAME = os.path.basename(file_path)

    file_size = os.path.getsize(file_path)
    print(f"Uploading '{file_path}' ({file_size / 1024 / 1024:.2f} MB) ...")

    with open(file_path, "rb") as f:
        if file_size <= CHUNK_SIZE:
            # Small file fallback
            dbx.files_upload(f.read(), f"{DROPBOX_SUBFOLDER_PATH}/{DROPBOX_FILE_NAME}", mode=WriteMode.add) #note, it uploads the file only if it is unique (doesn't exist already)
        else:
            # Start upload session
            session = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
            cursor = dropbox.files.UploadSessionCursor(session.session_id, f.tell())
            commit = dropbox.files.CommitInfo(path=f"{DROPBOX_SUBFOLDER_PATH}/{DROPBOX_FILE_NAME}", mode=WriteMode.add)
            
            while f.tell() < file_size:
                progress = f.tell() / file_size * 100
                print(f"Uploaded {progress:.2f}%")
                if (file_size - f.tell()) <= CHUNK_SIZE:
                    dbx.files_upload_session_finish(f.read(CHUNK_SIZE), cursor, commit)
                else:
                    dbx.files_upload_session_append_v2(f.read(CHUNK_SIZE), cursor)
                    cursor.offset = f.tell()

    print("Upload complete!")

if __name__ == "__main__":
    file_path = r"C:/Users/raine/Data/School/MIT/Freshman Year/UROP/CSV/2015-09-06.csv"

    upload_file(file_path)
