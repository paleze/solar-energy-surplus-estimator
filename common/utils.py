import os
import logging
from minio import Minio
from mlflow import mlflow, MlflowClient, MlflowException

# Get logger
app_logger = logging.getLogger()
app_logger.addHandler(logging.StreamHandler())
app_logger.setLevel(logging.INFO)



def get_secret_from_env_or_file(env_var=None, file_path_var=None):
    """
    Retrieve a secret value from an environment variable or a file.

    Args:
    - env_var (str): The name of the environment variable.
    - file_path_var (str): The name of the environment variable that contains the file path.

    Returns:
    - str: The secret value.
    """
    if env_var is not None:
        secret = os.getenv(env_var)
        if secret is not None:
            return secret

    if file_path_var is not None and os.path.exists(file_path_var):
        try:
            with open(file_path_var, 'r') as file:
                return file.read().strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read secret from file {file_path_var}: {e}") from e
    
    raise RuntimeError(f"Secret not found: neither {env_var} nor '{file_path_var}' is set.")



class MlflowMgr:

    """
    Mlflow manager class to handle Mlflow client and operations
    """

    mlflow_client =None
    # experiment_name=None
    # model_name=None
    # model_production_alias=None

    def __init__(self, mlflow_tracking_uri=None, experiment_name="solar-surplus-estimator-experiment", model_name="solar-surplus-estimator", model_production_alias="production"):

        """
        Initialize mlflow
        """

        self.mlflow_tracking_uri=mlflow_tracking_uri if not None else os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")
        self.experiment_name=experiment_name
        self.model_name=model_name
        self.model_production_alias=model_production_alias

    def get_client(self):

        """
        Return an Mlflow client instance
        """

        if self.mlflow_client is None:
            self.mlflow_client = MlflowClient(self.mlflow_tracking_uri)

        return self.mlflow_client


# Get Mlflow configuration and client
mlflow_mgr=MlflowMgr()
mlflow_client=mlflow_mgr.get_client()




class MinioMgr:

    """
    Minio manager class to handle Minio client and operations
    """

    app_logger = None
    s3_client = None


    def __init__(self, app_logger=None):

        """
        Initialize Minio
        """

        self.app_logger = app_logger
        self.access_key=get_secret_from_env_or_file("MINIO_ACCESS_KEY","/run/secrets/minio_root_user")
        self.secret_key=get_secret_from_env_or_file("MINIO_SECRET_KEY","/run/secrets/minio_root_password")
        self.minio_port=get_secret_from_env_or_file("MINIO_API_PORT")
        self.minio_region=get_secret_from_env_or_file("MINIO_REGION")
        self.minio_host=get_secret_from_env_or_file("MINIO_HOST")


    def log(self, message):
        if self.app_logger is not None:
            self.app_logger.info(message)

    def get_s3_client(self):
        
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

            self.log("S3 client created successfully")
        
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

            s3_client.fget_object(bucket_name=bucket_name, object_name=object_name, file_path=download_path)

            self.log(f"Downloaded {object_name} from bucket {bucket_name} to path {download_path}")

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

            s3_client.fput_object(
                bucket_name,
                object_name if object_name else os.path.basename(file_path),
                str(file_path)
            )
            self.log(f"Uploaded: {file_path} into bucket {bucket_name}")

        except Exception as e:
            self.log(f"Error uploading {file_path} into bucket {bucket_name}: {e}")
            raise e
    

minio_mgr=MinioMgr(app_logger)
s3_client = minio_mgr.get_s3_client()





