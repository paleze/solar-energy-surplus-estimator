
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

import os
import logging
from pathlib import Path
from cadwyn import endpoint
import requests
import pickle

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error 
import xgboost as xgb

from minio import Minio

import openmeteo_requests
import requests_cache
from retry_requests import retry

import mlflow



default_args = {
    'owner': 'airflow',
    'start_date': None,
}   


def get_secret_from_env_or_file(env_var, file_path_var=None):
    """
    Retrieve a secret value from an environment variable or a file.

    Args:
    - env_var (str): The name of the environment variable.
    - file_path_var (str): The name of the environment variable that contains the file path.

    Returns:
    - str: The secret value.
    """
    secret = os.getenv(env_var)
    if secret is not None:
        return secret

    if file_path_var is not None:
        file_path = os.getenv(file_path_var)
        if file_path is not None:
            try:
                with open(file_path, 'r') as file:
                    return file.read().strip()
            except Exception as e:
                raise RuntimeError(f"Failed to read secret from file {file_path}: {e}") from e
    
    raise RuntimeError(f"Secret not found: neither {env_var} nor {file_path_var} is set.")


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

    except Exception as e:
        raise RuntimeError("Failed to create Minio client: {error}".format(error=str(e))) from e
    
    logging.info("S3 client created successfully")
    
    return s3_client




def fecth_data(ti, pattern="*.csv", concat=True, add_filename_col=True, sort=True, **read_csv_kwargs) -> pd.DataFrame:
    """
    Download files from an S3 bucket and Load CSV files from `directory`.

    Args:
    - endpoint (str): url to folder containing CSV files
    - pattern (str): glob pattern to match files (default "*.csv").
    - concat (bool): if True, return a single concatenated DataFrame; if False, return a list of DataFrames.
    - add_filename_col (bool): if True, add a column '_source_file' with the filename for each row.
    - sort (bool): if True, sort files by name before loading.
    - **read_csv_kwargs: passed to `pd.read_csv` (e.g., parse_dates=["time"]).

    Returns:
    - pandas.DataFrame or list[pandas.DataFrame]
    """

    try:

        s3_client = get_s3_client()

        source_data_bucket=get_secret_from_env_or_file("MINIO_DATA_BUCKET")
        
        objects = s3_client.list_objects(bucket_name=source_data_bucket)
        
        # List all files in the bucket matching the pattern
        if objects is None:
           raise RuntimeError(f"No objects found in bucket {source_data_bucket} with pattern {pattern}")
        
        # Download each file
        downloadPath = Path("/tmp/solar_data_download")
        downloadPath.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading files to {downloadPath}")
        
        for obj in objects:
            s3_client.fget_object(bucket_name=source_data_bucket, object_name=obj.object_name, file_path=Path.joinpath(downloadPath, obj.object_name))
            

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from {endpoint}: {e}") from e

    # Load files from the local directory into a DataFrame
    
    files = list(downloadPath.glob(pattern))
    if sort:
        files = sorted(files)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            # re-raise with context
            raise RuntimeError(f"Failed to read {f}: {e}") from e
        if add_filename_col:
            df["_source_file"] = f.name
        dfs.append(df)

    if dfs.count==0:
        raise RuntimeError(f"Error creating data frame")

    if concat:
        dfs = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    logging.info(f"Fetched data frame has {len(df)} rows")

    return dfs

  

def transform_data(ti):

    """
    Manipulate features in raw data
    """

    df_raw = ti.xcom_pull(task_ids='fetch_data')

    # Create tranformed DataFrame from raw one
    df=pd.DataFrame()

    # Get time and month from timestamps
    timestamps = pd.to_datetime(df_raw["timestamp"], dayfirst=False, errors="raise")
    df["timestamp"]=timestamps
    df["time"]=timestamps.dt.time
    df["month"]=timestamps.dt.month

    # copy relevant columns from raw dataframe and create new ones
    df['solar_energy_exported'] = df_raw['solar_energy_exported']
    df['total_home_usage']=df_raw['consumer_energy_imported_from_grid']+df_raw['consumer_energy_imported_from_solar']+df_raw['consumer_energy_imported_from_battery']

    # Remove hours at early morning or late evening where no solar production can occur
    df = df[ ( (df.time.astype(str)>'06:00:00') & (df.time.astype(str)<'22:00:00') )]

    # compute solar surplus (at minimum it must be 0)
    df['solar_surplus']=df['solar_energy_exported'] - df['total_home_usage']
    df.solar_surplus=df.solar_surplus.apply(lambda x: max(0,x))

    # Get min and max dates
    start_date=min(df.timestamp).date()
    end_date=max(df.timestamp).date()

    logging.info(f"Transformed data frame has {len(df)} rows")
    
    return start_date, end_date, df



def get_historical_weather(ti):

    """
    Fetch historical weather data from Open-Meteo API according to source data date range
    """

    start_date, end_date, _ = ti.xcom_pull(task_ids='transform_data')

    # Get environment variables for location (default is old town Bergamo)
    latitude=float(get_secret_from_env_or_file("SITE_LATITUDE"))
    longitude=float(get_secret_from_env_or_file("SITE_LONGITUDE"))
    timezone=get_secret_from_env_or_file("SITE_TIMEZONE")

    logging.info(f"Parameters for historical weather forecast:")
    logging.info(f"- Date range: {start_date} - {end_date}")
    logging.info(f"- Location: Latitude {latitude}, Longitude {longitude}, Timezone {timezone}")
    

    try:

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
    
        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": {latitude},
            "longitude": {longitude},
            "start_date": {start_date},
            "end_date": { (end_date if end_date is not None else start_date) },
            "hourly": ["weather_code", "cloud_cover", "temperature"],
            "timezone": {timezone},
        }
        responses = openmeteo.weather_api(url, params=params)
        
        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        
        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        
        hourly_weather_code = hourly.Variables(0).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(1).ValuesAsNumpy()
        hourly_temperature = hourly.Variables(2).ValuesAsNumpy()
        
        hourly_data = {"timestamp": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = False),
            end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = False),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        
        hourly_data["weather_code"] = hourly_weather_code
        hourly_data["cloud_cover"] = hourly_cloud_cover
        hourly_data["temperature"] = hourly_temperature
        
        hourly_df = pd.DataFrame(data = hourly_data)

        logging.info(f"Retrieved {len(hourly_df)} hourly weather data points")
        
        return hourly_df
    
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch historical weather data: {e}") from e
    
    
    

def merge_historical_weather(ti):
    """
    Merge historical weather data with transformed source data
    """
    _, _, df = ti.xcom_pull(task_ids='transform_data')
    hourly_weather_df = ti.xcom_pull(task_ids='get_historical_weather')

    # merge energy data with weather historical data on timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=False)
    hourly_weather_df['timestamp'] = pd.to_datetime(hourly_weather_df['timestamp'], utc=False)
    df_merge = pd.merge(df, hourly_weather_df, how='left', on='timestamp')

    # fill missing data at half-hours with that from the beginning of the corresponding hours
    df_missing = df_merge[ ( np.isnan(df_merge.weather_code) | np.isnan(df_merge.cloud_cover) | np.isnan(df_merge.temperature) ) ]

    if df_missing.size>0:
        logging.info('Filling missing weather data at half-hours')
        df_merge = df_merge.reset_index()  # make sure indexes pair with number of rows
        
        for idx, row in df_merge.iterrows():    
            if row['timestamp'].minute==30:
                source_idx = ( idx-1 if idx > 0 else idx+1 )
                if (np.isnan(row['weather_code'])):
                    df_merge.at[ idx, 'weather_code'] = df_merge.loc[source_idx]['weather_code']
                if (np.isnan(row['cloud_cover'])):
                    df_merge.at[ idx, 'cloud_cover'] = df_merge.loc[source_idx]['cloud_cover']
                if (np.isnan(row['temperature'])):
                    df_merge.at[ idx, 'temperature'] = df_merge.loc[source_idx]['temperature']

    # Check whether some weather data is missing
    missing_count = df_merge[ ( np.isnan(df_merge['weather_code']) | np.isnan(df_merge['cloud_cover']) | np.isnan(df_merge['temperature']) )].size
    if missing_count>0:
        raise RuntimeError(f"After filling, still {missing_count} missing weather data points remain")

    logging.info(f"Merged data frame has {len(df_merge)} rows")

    return df_merge



def train_model(ti):

    """
    Train a model to estimate solar surplus based on weather data
    """
    
    df_merge = ti.xcom_pull(task_ids='merge_historical_weather')

    # Set features
    categorical=['time', 'month','weather_code']
    numerical=[ 'cloud_cover', 'temperature' ]

    # Convert categorical features to string type to prepare for DictVectorizer
    df_merge[categorical] = df_merge[categorical].astype(str)

    # Split merged data set 75% train / 25% test
    X_Train_df = df_merge.sample(frac = 0.75)
    X_Test_df = df_merge.drop(X_Train_df.index) 

    # Get a Dictionary Vectorizer to convert categorical features into new binary features (0-1)
    dv = DictVectorizer()

    # Get dictionaries of categorical features
    X_Train_dicts = X_Train_df[categorical+numerical].to_dict(orient='records')
    
    # Convert categorical features into a dataset which has as many columns as the label of each category
    X_train = dv.fit_transform(X_Train_dicts)

    # Same approach for test data set
    X_Test_dicts = X_Test_df[categorical+numerical].to_dict(orient='records')
    X_test = dv.fit_transform(X_Test_dicts)

    target = 'solar_surplus'
    y_train = X_Train_df[target].values
    y_test = X_Test_df[target].values

    # Set experiment tracking
    experiment_name="solar-surplus-experiment"
    mlflow_tracking_uri=get_secret_from_env_or_file("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # set temp local model path
    model_path=Path("/tmp/models")
    model_path.mkdir(parents=True, exist_ok=True)
    preprocessor_file=Path.joinpath(model_path,"preprocessor.b")

    logging.info("Start traininig model ...")
    with mlflow.start_run():
    
        X_Train_dm = xgb.DMatrix(X_train, label=y_train)
        X_test_dm = xgb.DMatrix(X_test, label=y_test)

        # Best params from offline hyperparameter tuning session
        best_params = {
            "learning_rate": 0.06442636015018084,
            "max_depth": 17,
            "min_child_weight": 11.057750768573072,
            "objective": "reg:squarederror",
            "reg_alpha": 0.07238551460828992,
            "reg_lambda": 0.06289038778129306,
            "seed": 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=X_Train_dm,
            num_boost_round=100,
            evals=[(X_test_dm, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(X_test_dm)
        rmse = root_mean_squared_error(y_test, y_pred)
        mlflow.log_metric("rmse_test", rmse)

        logging.info(f"Model training ended, rmse = {rmse}")

        with open(preprocessor_file, "wb") as f_out:
           pickle.dump(dv, f_out)

        mlflow.log_artifact(preprocessor_file, artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, name="solar-energy-surplus-predictor")




with DAG(
    dag_id='solar_energy_surplus_estimator_model_pipeline',
    default_args=default_args,
    start_date=None,
    catchup=False,
    tags=['solar-estimator'],
) as dag:
    
    fetch_data_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fecth_data,
        op_kwargs={
            "pattern": "*.csv",
            "concat": True,
            "add_filename_col": True,
            "sort": True,
            "read_csv_kwargs": {"parse_dates": ["timestamp"]},
        },
    )

    transform_data_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
    )

    get_historical_weather_task = PythonOperator(
        task_id='get_historical_weather',
        python_callable=get_historical_weather,
    )

    merge_historical_weather_task = PythonOperator(
        task_id='merge_historical_weather',
        python_callable=merge_historical_weather,
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

fetch_data_task >> transform_data_task >> get_historical_weather_task >> merge_historical_weather_task >> train_model_task