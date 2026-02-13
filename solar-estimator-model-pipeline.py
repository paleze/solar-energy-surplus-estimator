

from pathlib import Path
from cadwyn import endpoint
from dotenv import load_dotenv

import os
import logging
import requests

from minio import Minio

import openmeteo_requests
import requests_cache
from retry_requests import retry

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error 


import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials # use hyperpot package to perform parameter tuning
from hyperopt.pyll import scope


from mlflow import mlflow, MlflowClient

import pickle

import weather_data as wd


def get_secret_from_env_or_file(env_var, file_path_var=None, default=None):
    """
    Retrieve a secret value from an environment variable or a file.

    Args:
    - env_var (str): The name of the environment variable.
    - file_path_var (str): The name of the environment variable that contains the file path.
    - default (str): The default value to return if the environment variable and file path are not set.

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

    if default is not None:
        return default
    
    raise RuntimeError(f"Secret not found: neither {env_var} nor {file_path_var} is set.")


def get_s3_client():
    
    """
    Return a Minio client instance
    """
    
    try:

        access_key=get_secret_from_env_or_file("MINIO_ACCESS_KEY","/run/secrets/minio_root_user")
        secret_key=get_secret_from_env_or_file("MINIO_SECRET_KEY","/run/secrets/minio_root_password")
        minio_port=get_secret_from_env_or_file("MINIO_API_PORT",default=9080)
        minio_region=get_secret_from_env_or_file("MINIO_REGION", default="us-east-1")
        minio_host=get_secret_from_env_or_file("MINIO_HOST", default="localhost")

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


def fecth_data(s3_client, pattern="*.csv", concat=True, add_filename_col=True, sort=True, **read_csv_kwargs) -> pd.DataFrame:
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

        source_data_bucket=get_secret_from_env_or_file("MINIO_DATA_BUCKET",default="source-data")
        
        objects = s3_client.list_objects(bucket_name=source_data_bucket)
        
        # List all files in the bucket matching the pattern
        if objects is None:
           raise RuntimeError(f"No objects found in bucket {source_data_bucket} with pattern {pattern}")
        #elif len(objects) == 0:
           #raise RuntimeError(f"No objects found in bucket {source_data_bucket} with pattern {pattern}")
    
        #logging.info(f"Found {len(objects)} objects in bucket {source_data_bucket}")

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
            #df = pd.read_csv(f, **read_csv_kwargs)
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



def transform_data(df_raw: pd.DataFrame):

    """
    Manipulate features in raw data
    """

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
    df = df[ ( (df.time.astype(str)>='06:00:00') & (df.time.astype(str)<'21:00:00') )]

    # compute solar surplus (at minimum it must be 0)
    df['solar_surplus']=df['solar_energy_exported'] - df['total_home_usage']
    df.solar_surplus=df.solar_surplus.apply(lambda x: max(0,x))

    # Get min and max dates
    start_date=min(df.timestamp).date().strftime("%Y-%m-%d")
    end_date=max(df.timestamp).date().strftime("%Y-%m-%d")

    logging.info(f"Transformed data frame has {len(df)} rows")
    
    return start_date, end_date, df
        
    
def merge_historical_weather(df, hourly_weather_df):
    """
    Merge historical weather data with transformed source data
    """


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





def train_model(df_merge):

    """
    Train a model to estimate solar surplus based on weather data
    """

    # Set features
    categorical=['time', 'month','weather_code','cloud_cover' ]
    numerical=[ 'temperature' ]

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
    X_test = dv.transform(X_Test_dicts)

    target = 'solar_surplus'
    y_train = X_Train_df[target].values
    y_test = X_Test_df[target].values

    # Save data file
    #source_data_filename='data/source_data.csv'
    #df_merge.to_csv(source_data_filename, sep='\t', encoding='utf-8', index=False, header=True)

    experiment_name="solar-surplus-estimator-experiment"
    mlflow_tracking_uri=get_secret_from_env_or_file("MLFLOW_TRACKING_URI",default="http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # set temp local model path
    model_path=Path("/tmp/models")
    model_path.mkdir(parents=True, exist_ok=True)
    preprocessor_file=Path.joinpath(model_path,"preprocessor.b")

    # Enable auto logging
    mlflow.xgboost.autolog()
    
    logging.info("Start traininig model with hyperparameters optimization")

    X_Train_dm = xgb.DMatrix(X_train, label=y_train)
    X_test_dm = xgb.DMatrix(X_test, label=y_test)
    
    # tuning parameters (see https://hyperopt.github.io/hyperopt/ for reference )
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),  # explore solutions from exp(-3) and exp(0)
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:squarederror',
        'seed': 42
    }

    def hyperparameter_objective(params):

        with mlflow.start_run():

            model_type="xgboost"
            model_name=experiment_name+"_"+model_type
            mlflow.set_tag("model_type", model_type)
            mlflow.log_params(params)

            mlflow.xgboost.autolog()
            booster = xgb.train(
                params=params,
                dtrain=X_Train_dm,
                num_boost_round=1000, # Maximum number of iterations for hyperparameter tuning
                evals=[(X_test_dm, 'validation')],
                early_stopping_rounds=50 # Stop after 50 iterations with no improvement
            )
            y_pred = booster.predict(X_test_dm)
            rmse = root_mean_squared_error(y_test, y_pred)
            mlflow.log_metric("rmse_test", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    # Start the hyperparameter tuning
    best_result = fmin(
        fn=hyperparameter_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20, # number of hyperparameters tuning runs
        trials=Trials()
    )


    # Get best model
    best_run_id, best_run_rmse, best_model_id = get_best_model()

    logging.info(f"Model training ended, best_run_id = {best_run_id}, best rmse = {best_run_rmse}, best model id = {best_model_id}")

    # Save preprocessor DictionaryVectorzizer as a best run artifact
    with open(preprocessor_file, "wb") as f_out:
      pickle.dump(dv, f_out)


    mlflow.log_artifact(preprocessor_file, artifact_path="preprocessor", run_id=best_run_id)
    

def get_best_model():

    runs = mlflow.search_runs(filter_string="status='FINISHED'")
    best_run = runs.loc[ runs['metrics.rmse_test'].idxmin() ]
    best_run_id = best_run['run_id']
    best_run_rmse = best_run['metrics.rmse_test']
    

    model_filter_string=f"source_run_id='{best_run_id}'"
    models = mlflow.search_logged_models(filter_string=model_filter_string)
    best_model_id=models.at[0,"model_id"]

    logging.info(f'best run_id: {best_run_id}, rmse: {best_run_rmse}, best_model_id: {best_model_id}')
    
    return best_run_id, best_run_rmse, best_model_id


def register_and_alias_best_model():
    
    
    experiment_name="solar-surplus-estimator-experiment"
    mlflow_tracking_uri=get_secret_from_env_or_file("MLFLOW_TRACKING_URI",default="http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Get best model
    best_run_id, best_run_rmse, best_model_id = get_best_model()

    # Get MlFlow Client
    client = MlflowClient(mlflow_tracking_uri)

    model_name = "solar-surplus-estimator"
    model_version_alias = "production"

    try:

        model_info = client.get_model_version_by_alias(model_name, model_version_alias)

    except mlflow.exceptions.RestException as e:
        # Model name or alias not found 
        model_info = None
    
    if model_info is not None and model_info.run_id==best_run_id:

        best_version = model_info.version
        logging.info(f"Current model {model_name} with version {model_info.version}  is still the best one")
    
    else:

        # register and set alias to a new model
        model_uri = f"runs:/{best_run_id}/model"
        best_version = mlflow.register_model(model_uri, model_name).version
    
        client.set_registered_model_alias(
            model_name, model_version_alias, best_version  
        )

        logging.info(f"registered model {model_name} with new version {best_version} as alias {model_version_alias}")
    
    return { 
            "best_run_id": best_run_id,
            "best_run_rmse": best_run_rmse,
            "best_model_id": best_model_id,
            "best_version": best_version
    }


def main():

    # Load environment variables from local .env file
    load_dotenv(".env")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    s3_client = get_s3_client()
    
    df = fecth_data(s3_client, pattern="*.csv", concat=True, add_filename_col=True, sort=True,op_kwargs={
            "pattern": "*.csv",
            "concat": True,
            "add_filename_col": True,
            "sort": True,
            "read_csv_kwargs": {"parse_dates": ["timestamp"]},
        },
    )
    
    start_date, end_date, df_transformed = transform_data(df)


    latitude=float(get_secret_from_env_or_file("SITE_LATITUDE",default="45.704178167781265"))
    longitude=float(get_secret_from_env_or_file("SITE_LONGITUDE",default="9.663174137924273"))
    timezone=get_secret_from_env_or_file("SITE_TIMEZONE", default="Europe/Berlin")

    hourly_weather_df = wd.get_weather_data(start_date, end_date, latitude, longitude, timezone)


    df_merge = merge_historical_weather(df_transformed, hourly_weather_df)

    train_model(df_merge)

    
    res=register_and_alias_best_model()
    logging.info(f"Model training completed - best_run_id: {res["best_run_id"]}, best rmse: {res["best_run_rmse"]}, best model id: {res["best_model_id"]}, best version: {res["best_version"]}")


if __name__ == "__main__":
    main()
    