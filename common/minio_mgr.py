import os

from minio import Minio


import utils

class MinioMgr:

    """
    Minio manager class to handle Minio client and operations
    """

    app_logger = utils.app_logger
    s3_client = None


    def __init__(self):

        """
        Initialize Minio
        """

        self.access_key=utils.get_secret_from_env_or_file("MINIO_ACCESS_KEY","/run/secrets/minio_root_user")
        self.secret_key=utils.get_secret_from_env_or_file("MINIO_SECRET_KEY","/run/secrets/minio_root_password")
        self.minio_port=utils.get_secret_from_env_or_file("MINIO_API_PORT")
        self.minio_region=utils.get_secret_from_env_or_file("MINIO_REGION")
        self.minio_host=utils.get_secret_from_env_or_file("MINIO_HOST")

        self.s3_client = self.get_client()

    def get_client(self):
        
        """
        Return a Minio client instance
        """
        
        try:

            if self.s3_client is not None:
                return self.s3_client
        

            # Create client with access key and secret key with specific region.
            self.s3_client = Minio(
                "{host}:{port}".format(host=self.minio_host,port=self.minio_port),
                access_key=self.access_key,
                secret_key=self.secret_key,
                region=self.minio_region,
                secure=False
            )

            self.app_logger.info("S3 client created successfully")
        
            return self.s3_client

        except Exception as e:
            raise RuntimeError("Failed to create Minio client: {error}".format(error=str(e))) from e


    def get_file_from_bucket(self, bucket_name, download_path, object_name=None):
    
        """
        Download a file from a S3 bucket.
            Args:
                bucket_name: Name of the S3 bucket
                object_name: Name of the object in the bucket
                download_path: Local path to save the downloaded file
        """

        try:
            
            if not object_name:
                object_name = os.path.basename(download_path)

            self.s3_client.fget_object(bucket_name=bucket_name, object_name=object_name, file_path=download_path)

            self.app_logger.info(f"Downloaded {object_name} from bucket {bucket_name} to path {download_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to download {object_name} from bucket {bucket_name}: {e}") from e
    


    def upload_to_bucket(self, file_path, bucket_name, object_name=None):
    
        """
          Upload data to Minio bucket

            Args:
                file_path: Local path of the file to be uploaded
                bucket_name: Name of the S3 bucket
                object_name: Name of the object in the bucket (optional, defaults to file name)
        """
        
        try:

            self.s3_client.fput_object(
                bucket_name,
                object_name if object_name else os.path.basename(file_path),
                str(file_path)
            )
            
            self.app_logger.info(f"Uploaded: {file_path} into bucket {bucket_name}")

        except Exception as e:
            self.app_logger.error(f"Error uploading {file_path} into bucket {bucket_name}: {e}")
            raise e