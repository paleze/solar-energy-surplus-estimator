import os
import logging
from minio import Minio
from mlflow import mlflow, MlflowClient, MlflowException

# Get logger
app_logger = logging.getLogger()
app_logger.addHandler(logging.StreamHandler())
app_logger.setLevel(logging.INFO)


# Airflow data
model_training_pipeline_dag_id="solar_energy_surplus_estimator_model_pipeline"

class MlflowConfig:

    mlflow_client = None

    def __init__(self):

        """
        Initialize mlflow
        """

        self.mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")
        self.experiment_name="solar-surplus-estimator-experiment"
        self.model_name="solar-surplus-estimator"
        self.model_production_alias="production"

    def get_client(self):
        if self.mlflow_client is None:
            self.mlflow_client = MlflowClient(self.mlflow_tracking_uri)

        return self.mlflow_client


# Get Mlflow configuration and client
mlflow_config=MlflowConfig()
mlflow_client=mlflow_config.get_client()


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


def get_s3_client():
    
    """
    Return a Minio client instance
    """
    
    try:
         
        access_key=get_secret_from_env_or_file("MINIO_ACCESS_KEY","/run/secrets/minio_root_user")
        secret_key=get_secret_from_env_or_file("MINIO_SECRET_KEY","/run/secrets/minio_root_password")
        minio_port=get_secret_from_env_or_file("MINIO_API_PORT")
        minio_region=get_secret_from_env_or_file("MINIO_REGION")
        minio_host=get_secret_from_env_or_file("MINIO_HOST")

        # Create client with access key and secret key with specific region.
        s3_client = Minio(
            "{host}:{port}".format(host=minio_host,port=minio_port),
            access_key=access_key,
            secret_key=secret_key,
            region=minio_region,
            secure=False
        )

        logging.info("S3 client created successfully")
    
        return s3_client

    except Exception as e:
        raise RuntimeError("Failed to create Minio client: {error}".format(error=str(e))) from e

# Get s3 client
s3_client = get_s3_client()

def get_file_from_bucket(bucket_name, download_path, object_name=None):
    
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

        logging.info(f"Downloaded {object_name} from bucket {bucket_name} to path {download_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to download {object_name} from bucket {bucket_name}: {e}") from e
    

def upload_to_bucket(file_path, bucket_name, object_name=None):
    
    # Upload training data set to Minio bucket
    
    try:

        s3_client.fput_object(
            bucket_name,
            object_name if object_name else os.path.basename(file_path),
            str(file_path)
        )
        logging.info(f"Uploaded: {file_path} into bucket {bucket_name}")

    except Exception as e:
        logging.info(f"Error uploading {file_path} into bucket {bucket_name}: {e}")
        raise e









