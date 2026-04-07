from minio import Minio
from pathlib import Path
import os
import argparse
from dotenv import load_dotenv


def upload_files_to_minio(minio_client, local_folder, bucket_name, pattern="*.csv", prefix=""):
    """
    Upload all files from a local folder to a Minio server.
    
    Args:
        minio_client: Minio client object
        local_folder: Path to local folder containing files
        bucket_name: Name of the Minio bucket
        prefix: Optional prefix for objects in the bucket
    """
    folder_path = Path(local_folder)
    
    if not folder_path.exists():
        raise ValueError(f"Folder not found: {local_folder}")
    
    print(f"Uploading files from {local_folder} to bucket '{bucket_name}' with prefix '{prefix}'")
    for file_path in folder_path.glob(pattern):
        if file_path.is_file():
            object_name = f"{prefix}{file_path.name}".lstrip("/")
            
            try:
                minio_client.fput_object(
                    bucket_name,
                    object_name,
                    str(file_path)
                )
                print(f"Uploaded: {file_path.name}")
            except Exception as e:
                print(f"Error uploading {file_path.name}: {e}")

def main():

    # Get source folder by parsing program arguments
    parser = argparse.ArgumentParser(description="Upload data files to Minio")
    parser.add_argument("--folder", default=".", help="Local folder to upload (default: .)")
    args = parser.parse_args()
    local_folder = args.folder
    
    
    # Load environment variables
    load_dotenv(".env")

    minio_port = os.getenv("MINIO_HOST_API_PORT", "9000")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minio")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minio123")
    minio_source_bucket= os.getenv("MINIO_SOURCE_DATA_BUCKET", "source-data")

    # Initialize Minio client
    client = Minio(
        f"localhost:{minio_port}",
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )
    
    upload_files_to_minio(client, local_folder , minio_source_bucket)


if __name__ == "__main__":
    main()
    