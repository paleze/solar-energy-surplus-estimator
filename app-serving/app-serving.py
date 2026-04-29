from datetime import date, datetime, timedelta, timezone
import os
import time
import logging
from logging import getLogger
import json
import pickle


# UI
import streamlit as st

# Data processing and model prediction
import numpy as np
import pandas as pd
import xgboost as xgb

# Mlflow
from mlflow import MlflowException


# Drift analysis
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

# Metrics
import metrics
from streamlit_extras.prometheus import streamlit_registry

# Local files
import utils
import weather_data as wd
from airflow_mgr import AirflowMgr
from mlflow_mgr import MlflowMgr
from minio_mgr import MinioMgr



# Set evaluation features 
categorical_features=['time', 'month','weather_code', 'cloud_cover']
numerical_features=[ 'temperature' ]
target_feature = 'solar_surplus'

# Set drift analysis features
drift_categorical_features = ['weather_code', 'cloud_cover']
drift_numerical_features = numerical_features

analytics_data_bucket=utils.get_secret_from_env_or_file("MINIO_ANALYTICS_DATA_BUCKET")
prediction_results_filename = utils.get_secret_from_env_or_file("MINIO_PREDICTION_RESULTS_FILE_NAME")
prediction_results_path=f"/tmp/{prediction_results_filename}"

training_data_filename = utils.get_secret_from_env_or_file("MINIO_TRAINING_DATA_FILE_NAME")
training_data_path=f"/tmp/{training_data_filename}"


DEFAULT_DRIF_THRESHOLD = 1.0 # default threshold for data drift detection, to be used in evidently report (e.g. for Kolmogorov-Smirnov test)
PREDICTION_DATA_SIZE_DRIFT_ANALYSIS_THRESHOLD = 50 # minimum number of prediction data points required to perform drift analysis

app_logger = utils.app_logger


# Get Airflow Manager instance
airflow_mgr=AirflowMgr()

# Get Mlflow manager instance
mlflow_mgr=MlflowMgr()

# Get S3 manager instance
s3_mgr=MinioMgr()


@st.cache_resource
def get_model_artifacts():

    """
    Return the production model artifacts to be used for prediction
    """
    try:

        # get a model version by alias
        prod_version = None

        try:
            prod_version = mlflow_mgr.get_production_model_version()
        except MlflowException as e:
            pass

        if prod_version is None:
            # No production model version found, trigger the model training pipeline
            dag_run_id = airflow_mgr.trigger_airflow_dag()
            raise RuntimeError(f"No version found for model '{mlflow_mgr.model_name}' with alias '{mlflow_mgr.model_production_alias}'. Triggered model training pipeline '{airflow_mgr.model_training_pipeline_dag_id}' with dag run id: {dag_run_id}, follow pipeline progress in Airflow UI")

        # Load model
        booster = mlflow_mgr.get_production_model()
        
        # Load data preprocesser DictionaryVectorizer
        path = mlflow_mgr.client.download_artifacts(run_id=prod_version.run_id, path="preprocessor/preprocessor.b")

        with open(path, 'rb') as f_out:
            dv = pickle.load(f_out)
        
        return booster, dv
    
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifact: {e}") from e



def save_prediction_results(df):
    """
    Save prediction results to a csv file for later drift analysis
    
    :param df: DataFrame containing the prediction results to be saved
    """

    try:

        if not os.path.exists(prediction_results_path):
            try:
                s3_mgr.get_file_from_bucket(analytics_data_bucket, prediction_results_path)
            except Exception as e:
                app_logger.error(f"Failed to download prediction results file: {e}, creating a new one")

        # Update local prediction results file with new predictions
        df.to_csv(prediction_results_path, sep='\t', encoding='utf-8', index=False, header=(not os.path.exists(prediction_results_path)), mode='a')        

        # Upload updated prediction results file to S3 bucket    
        s3_mgr.upload_to_bucket(prediction_results_path, analytics_data_bucket)

    except Exception as e:
        raise RuntimeError(f"Failed to save prediction results: {e}") from e
    


def predict(df_eval):

    """
    Predicts the solar energy surplus generated along the day
    
    :param df_eval: weather forecast data for the rest of the day
    :return: predicted solar energy surplus (Wh)
    :rtype: float
    """

    try:
        
        # Get production model artifacts
        model, dv = get_model_artifacts()

        # Get prediction start time for latency monitoring
        start_time = time.time()

        # Apply DictionaryVectorizer
        df_eval[categorical_features] = df_eval[categorical_features].astype(str)
        X_eval_dicts = df_eval[categorical_features + numerical_features].to_dict(orient='records')
        X_eval = dv.transform(X_eval_dicts)

        # Model preriction
        X_eval_dm = xgb.DMatrix(X_eval)
        y_pred = model.predict(X_eval_dm)

        # Remove spurious negative surplus predictions
        z = np.zeros(len(y_pred))
        y_pred_bottomed = np.maximum(y_pred,z)
        solar_surplus_prediction = sum(y_pred_bottomed)

        # Record metrics
        curr_time=df_eval["time"].iloc[0]
        curr_month=df_eval["month"].iloc[0]
        metrics.prediction_latency.labels(time=curr_time, month=curr_month).observe(time.time() - start_time)
        metrics.prediction_counter.labels(time=curr_time, month=curr_month).inc()
        
        app_logger.info(f"Solar surplus predictions: {y_pred}")
        app_logger.info(f"Solar surplus predictions (zero-bottmoed): {y_pred_bottomed}")
        app_logger.info(f"Total solar surplus prediction for the rest of day: {solar_surplus_prediction}")

        # Append prediction results to csv file for later drift analysis
        save_prediction_results(df_eval)
        
        # create a dataframe with predictions to be displayed
        df_surplus = pd.DataFrame(y_pred_bottomed, columns=[target_feature])
        df_pred = df_eval.reset_index(drop=True).join(df_surplus)

        return  df_pred, solar_surplus_prediction
    
    except Exception as e:
        raise RuntimeError(f"Failed to perform model predictions: {e}") from e



def get_evaluation_data(evaluation_date, evaluation_time):

    """
    Return evaluation data to be used for model prediction, including weather forecast and features engineering
    
    :param evaluation_date: Date for which the weather forecast is retrieved and features are computed
    :param evaluation_time: Intial time of the day from which the weather forecast is retrieved and features are computed
    """

    latitude=float(os.getenv("SITE_LATITUDE", "45.704178167781265"))
    longitude=float(os.getenv("SITE_LONGITUDE", "9.663174137924273"))
    timezone=os.getenv("SITE_TIMEZONE", "Europe/Berlin")
    
    # Input data processing
    evaluation_hour=int(evaluation_time.strftime("%H"))
    evaluation_minute=int(evaluation_time.strftime("%M"))
    weather_forecast_hour = evaluation_hour if evaluation_minute<30 else (evaluation_hour+1)
    weather_forecast_time=f"{weather_forecast_hour:02}:00:00"
    evaluation_date_str = evaluation_date.strftime("%Y-%m-%d")
    
    # Get weather forecast
    df_wf_hourly = wd.get_weather_data(start_date=evaluation_date_str, end_date=evaluation_date_str,latitude=latitude,longitude=longitude, timezone=timezone)
    app_logger.info("Weather forecast hourly data\n", df_wf_hourly)

    # Features engineering
    df_eval=pd.DataFrame()
    weather_fields=["weather_code","cloud_cover","temperature"]
    df_eval["time"]=df_wf_hourly.timestamp.dt.time
    df_eval["month"] = evaluation_date.month
    df_eval[weather_fields] = df_wf_hourly[weather_fields]

    # keep a reasonable solar production time window
    df_eval = df_eval[ ( (df_eval.time.astype(str)>=weather_forecast_time) & (df_eval.time.astype(str)<'20:30:00') ) ] 
    
    # Replicate weather data at half-hours to match model data
    df_eval_30 = df_eval.copy()
    today = date.today() # can be any date
    for idx, row in df_eval_30.iterrows():
        dt = datetime.combine(today, row["time"]) + timedelta(minutes=30)
        df_eval_30.at[idx,"time"] = dt.time()

    df_eval = pd.concat([df_eval, df_eval_30]).sort_values('time')

    return df_eval




def perform_drift_analysis(minimum_data_size: int, drift_threshold:float):
    
    """
    Perform drift analysis comparing the distribution of the latest predictions with the distribution of previous predictions, and display results in Streamlit
    
    :param prediction_results_filename: path to csv file where the latest prediction results have been stored, and with the history of previous predictions
    """

    # download prediction results file from S3 bucket if it does not exist locally
    try:

        if not os.path.exists(prediction_results_path):
           s3_mgr.get_file_from_bucket(analytics_data_bucket, prediction_results_path)

    except Exception as e:
       raise RuntimeError(f"Failed to get past prediction data for drift analysis, please perform some surplus estimation first.")
    
    try:
        
        
        # Load prediction results data
        prediction_df = pd.read_csv(prediction_results_path, sep='\t', encoding='utf-8')

        if prediction_df.size < minimum_data_size:
            st.warning("⚠️ Not enough reference data for drift analysis")
            return

        # download training data file from S3 bucket if it does not exist locally
        if not os.path.exists(training_data_path):
            s3_mgr.get_file_from_bucket(analytics_data_bucket, training_data_path)

        # Load training data set to be used as reference data for drift analysis
        reference_df = pd.read_csv(training_data_path, sep='\t', encoding='utf-8')

        # Create evidently Dataset oobjects
        data_definition = DataDefinition(
            numerical_columns=drift_numerical_features,
            categorical_columns=drift_categorical_features
        )

        current_df=prediction_df[-minimum_data_size:]
        current_data = Dataset.from_pandas(
            current_df,
            data_definition=data_definition
        )

        reference_data = Dataset.from_pandas(
            reference_df,
            data_definition=data_definition
        )

        # Create and run evidently report for data drift analysis
        report = Report(metrics=[DataDriftPreset(threshold=drift_threshold)])

        # Generate Column-Specific Reports
        drift_report = report.run(reference_data=reference_data, current_data=current_data)
        drift_json = json.loads(drift_report.json())

        drift_detected=False

        # Update drift metrics
        for metric in drift_json["metrics"]:

            if not metric["metric_name"].startswith("ValueDrift"):
                continue

            config = metric["config"]
            threshold=config["threshold"]
            
            value=metric["value"]
            if value > threshold:
                drift_detected=True
                metrics.drift_score_gauge.labels(feature=config["column"], method=config["method"], threshold=threshold).set(metric["value"])    
                metrics.drift_feature_counter.labels(feature=config["column"], method=config["method"], threshold=threshold).inc()
                metrics.drift_threshold_gauge.labels(feature=config["column"], method=config["method"]).set(threshold)
        
        # Update feature distribution metrics
        feature_columns= drift_numerical_features + drift_categorical_features
        feature_mean = current_df[feature_columns].mean()
        feature_std = current_df[feature_columns].std()
        
        for column in feature_columns:
            metrics.feature_mean_gauge.labels(feature=column).set(feature_mean[column])
            metrics.feature_std_gauge.labels(feature=column).set(feature_std[column])

        return drift_report, drift_detected
    
    except Exception as e:
        st.error(f"⚠️ Error: Failed to perform drift analysis: {e}")



def main():

    try:

        # Get environment variables
        battery_full_capacity = float(os.getenv("BATTERY_CAPACITY_KWH",13.5))
        
        # Get current datetime
        current_dateTime = datetime.now()
        current_date=current_dateTime.strftime("%Y-%m-%d")
        current_time=current_dateTime.strftime("%H:%M")

        # Streamlit UI
        st.set_page_config(page_title="Solar Energy Surplus Estimator", layout="wide")
        st.title("☀️ Solar Energy Surplus Estimator")

        # Input section
        st.header("Input Features")
        col1, col2, col3, col4 = st.columns(4, border=True)
        evaluation_date = col1.date_input("Date (YYYY/MM/dd):", width=200, value=current_date, min_value=date(2020, 8, 1), max_value=current_date)
        evaluation_time = col2.time_input("Start time (HH:MM):", width=200, value=current_time)
        battery_full_capacity = col3.number_input("Battery capacity (KWh):", 
            width=200, min_value=0.0, max_value=1000.0, step=0.1, value=battery_full_capacity, format="%.1f")
        battery_usage_percentage = col4.number_input("Battery capacity usage (%):", width=200, min_value=0, max_value=100)
        
        # Make prediction
        if st.button("Estimate Surplus", type="primary"):

            st.info(f"Downloading weather data...")

            # Process input data to be used for model prediction
            df_eval = get_evaluation_data(evaluation_date, evaluation_time)

            st.info("Performing prediction...")

            # Compute model prediction
            df_pred, solar_surplus_prediction = predict(df_eval)
            solar_surplus_prediction = solar_surplus_prediction/1000 # prediction is returned in Wh

            with st.expander("Prediction data preview", expanded=True):
                st.dataframe(df_pred.style.format({f"{target_feature}": "{:.1f}".format, "temperature": "{:.1f}".format}),
                    width=700, hide_index=True)

            st.info("Computing results...")

            # Compute solar energy surplus that will exceed battery residual capacity
            battery_residual_capacity = float(battery_full_capacity * (100 - battery_usage_percentage)/100)
            solar_surplus_to_grid = max(0, solar_surplus_prediction - battery_residual_capacity)
            battery_predicted_usage = round(min(100, battery_usage_percentage + (solar_surplus_prediction/battery_full_capacity*100) ))

            # Display results
            st.success(f"✅ Prediction ready")
            st.subheader("Prediction Results")
            col1, col2, col3 = st.columns(3, border=True)
            col1.metric("Predicted solar energy surplus generated until sunset (KWh): ", f"{solar_surplus_prediction:.2f}")
            col2.metric("Predicted battery capacity usage at sunset (including stored solar surplus) (%): ", f"{battery_predicted_usage}")
            col3.metric("Predicted solar energy surplus exported to the grid (KWh):", f"{solar_surplus_to_grid:.2f}")

        # Data drift section
        st.header("Data Drift Analysis")
        col1, col2 = st.columns(2, border=True, gap="small",width=500)
        prediction_data_size_threshold = col1.number_input("Predicted samples threshold:",
            width=200, min_value=0, value=PREDICTION_DATA_SIZE_DRIFT_ANALYSIS_THRESHOLD)
        drift_threshold = col2.number_input("Drift Score threshold:", 
            width=200, min_value=0.0, value=DEFAULT_DRIF_THRESHOLD,max_value=10.0, step=0.25)

        # Perform drift analysis and display results
        if st.button("Perform Data Drift analysis", type="primary"):
             
            drift_report, drift_detected = perform_drift_analysis(minimum_data_size=prediction_data_size_threshold, drift_threshold=drift_threshold)

            with st.expander("Data Drift Analysis", expanded=True):
                st.write("The following report compares the distribution of the latest predictions with the distribution of reference model training data to detect potential data drift:")
                st.components.v1.html(drift_report.get_html_str(as_iframe=False), height=800, scrolling=True)

            if drift_detected:
                try:
                    st.warning(f"⚠️ Data drift detected! A model retraining needs to be triggered")
                    dag_run_id = airflow_mgr.trigger_airflow_dag()
                    st.info(f"✅ Triggered model retraining with dag run id: {dag_run_id}, follow pipeline progress in Airflow UI")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
            else:
                st.success("✅ No data drift detected. Set a lower Drift Score threshold according to the detected Drift Scores to force drift detection and trigger model retraining")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
    

if __name__=="__main__":
    main()






