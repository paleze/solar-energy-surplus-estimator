

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
#from airflow.providers.http.hooks.http import HttpHook
#from airflow.providers.postgres.hooks.postgres import PostgresHook
#from airflow.decorators import task
#from airflow.utils.dates import days_ago
from pathlib import Path
from cadwyn import endpoint
#from dotenv import load_dotenv
import requests
#import json
import os
import logging
import pandas as pd
import numpy as np


from minio import Minio

global_s3_client = None

default_args = {
    'owner': 'airflow',
    'start_date': None,
}   


def get_s3_client():
    
    """
    Return a Minio client instance
    """

    if global_s3_client is not None:
        return global_s3_client
    
    try:

        access_key=os.getenv("MINIO_ACCESS_KEY")
        secret_key=os.getenv("MINIO_SECRET_KEY")
        minio_port=os.getenv("MINIO_HOST_API_PORT",9000)
        minio_region=os.getenv("MINIO_REGION","us-est-1")

        # Create client with access key and secret key with specific region.
        global_s3_client = Minio(
            "localhost:{port}".format(port=minio_port),
            access_key=access_key,
            secret_key=secret_key,
            region=minio_region,
        )

    except Exception as e:
        raise RuntimeError("Failed to create Minio client: {error}".format(error=str(e))) from e
    
    return global_s3_client


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

        s3_client = ti.xcom_pull(task_ids='get_s3_client')
        
        source_data_bucket=os.getenv("MINIO_DATA_BUCKET","source-data")
        
        objects = s3_client.list_objects(bucket_name=source_data_bucket, object_name=pattern)

        # List all files in the bucket matching the pattern
        if objects is None:
           raise RuntimeError(f"No objects found in bucket {source_data_bucket} with pattern {pattern}")
        elif objects.size == 0:
           raise RuntimeError(f"No objects found in bucket {source_data_bucket} with pattern {pattern}")
    
        # Download each file to the current directory
        for obj in objects:
            s3_client.fget_object(bucket_name=source_data_bucket, object_name=obj.object_name, file_path=obj.object_name)

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from {endpoint}: {e}") from e

    # Now load the files from the local directory into a DataFrame
    p = Path(".")
    files = list(p.glob(pattern))
    if sort:
        files = sorted(files)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, **read_csv_kwargs)
        except Exception as e:
            # re-raise with context
            raise RuntimeError(f"Failed to read {f}: {e}") from e
        if add_filename_col:
            df["_source_file"] = f.name
        dfs.append(df)

    if dfs.count==0:
        raise RuntimeError(f"Error creating data frame")

    if concat:
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
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
    logging.info(f"Date range for historical weather forecast: {min_date} - {max_date}")

    return start_date, end_date, df
    


def get_historical_weather(ti):

    """
    Fetch historical weather data from Open-Meteo API according to source data date range
    """

    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    start_date, end_date, _ = ti.xcom_pull(task_ids='transform_data')

    # Get environment variables for location
    latitude=float(os.getenv("SITE_LATITUDE"))
    longitude=float(os.getenv("SITE_LONGITUDE"))
    timezone=os.getenv("SITE_TIMEZONE","Europe/Berlin")

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


with DAG(
    dag_id='solar_energy_surplus_estimator_model_pipeline',
    default_args=default_args,
    start_date=None,
    schedule_interval=None,
    catchup=False,
    tags=['solar', 'energy', 'surplus', 'estimation'],
) as dag:
    
    get_s3_client_task = PythonOperator(
        task_id='get_s3_client',
        python_callable=get_s3_client,
    )

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

    get_s3_client_task >> fetch_data_task >> transform_data_task >> get_historical_weather_task >> merge_historical_weather_task