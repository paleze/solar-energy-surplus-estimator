import streamlit as st


import numpy as np
import pandas as pd
import xgboost as xgb


from datetime import date, datetime, timedelta
import os
import logging
import pickle

from mlflow import mlflow, MlflowClient

import weather_data as wd



def init_mlflow():

    """
    Initialize mlflow
    """

    global mlflow_tracking_uri
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")

    global experiment_name
    experiment_name="solar-surplus-estimator-experiment"

    global model_name
    model_name = "solar-surplus-estimator"

    global model_production_alias
    model_production_alias = "production"

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Get MlFlow Client
    global client
    client = MlflowClient(mlflow_tracking_uri)



@st.cache_resource
def get_model_artifacts():

    """
    Return the production model artifacts to be used for prediction
    """
    try:

        # get a model version by alias
        mlflow_version = client.get_model_version_by_alias(model_name, model_production_alias)

        if mlflow_version is None:
             raise RuntimeError(f"No version found for model '{model_name}' with alias '{model_production_alias}', run the model pipeline to create one")

        # Load model
        booster = mlflow.xgboost.load_model(f"models:/{model_name}@{model_production_alias}")
        
        # Load data preprocesser DictionaryVectorizer
        path = client.download_artifacts(run_id=mlflow_version.run_id, path="preprocessor/preprocessor.b")

        with open(path, 'rb') as f_out:
            dv = pickle.load(f_out)
        
        return booster, dv
    
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifact: {e}") from e    



def predict(df_eval) -> float:

    """
    Predicts the solar energy surplus generated along the day
    
    :param df_eval: weather forecast data for the rest of the day
    :return: predicted solar energy surplus (Wh)
    :rtype: float
    """

    try:
         
        # Get production model artifacts
        model, dv = get_model_artifacts()

        # Set evaluation features 
        categorical=['time', 'month','weather_code', 'cloud_cover']
        numerical=[ 'temperature' ]

        # Apply DictionaryVectorizer
        df_eval[categorical] = df_eval[categorical].astype(str)
        X_eval_dicts = df_eval[categorical+numerical].to_dict(orient='records')
        X_eval = dv.transform(X_eval_dicts)

        # Model preriction
        X_eval_dm = xgb.DMatrix(X_eval)
        y_pred = model.predict(X_eval_dm)

        # Remove spurious negative surplus predictions
        z = np.zeros(len(y_pred))
        y_pred_bottomed = np.maximum(y_pred,z)
        solar_surplus_prediction = sum(y_pred_bottomed)

        logging.info(f"Solar surplus predictions: {y_pred}")
        logging.info(f"Solar surplus predictions (zero-bottmoed): {y_pred_bottomed}")
        logging.info(f"Total solar surplus prediction for the rest of day: {solar_surplus_prediction}")
        
        return solar_surplus_prediction
    
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
    logging.info("Weather forecast hourly data\n", df_wf_hourly)

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


def main():

    try:

        # Initialize mlflow
        init_mlflow()

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
        evaluation_date = st.date_input("Date (YYYY/MM/dd):", value=current_date)
        evaluation_time = st.time_input("Start time (HH:MM):", value=current_time)
        battery_full_capacity = st.number_input("Battery capacity (KWh):", min_value=0.0, step=0.1, value=battery_full_capacity, format="%.1f")
        battery_usage_percentage = st.number_input("Battery capacity usage (%):", min_value=0, max_value=100)
        
        # Make prediction
        if st.button("Estimate Surplus", type="primary"):

            st.info(f"Downloading weather data...")

            # Process input data to be used for model prediction
            df_eval = get_evaluation_data(evaluation_date, evaluation_time)
     
            st.info("Performing prediction...")

            # Compute model prediction
            solar_surplus_prediction = predict(df_eval)/1000 # prediction is returned in Wh

            st.info("Computing results...")

            # Compute solar energy surplus that will exceed battery residual capacity
            battery_residual_capacity = float(battery_full_capacity * (100 - battery_usage_percentage)/100)
            solar_surplus_to_grid = max(0, solar_surplus_prediction - battery_residual_capacity)
            battery_predicted_usage = round(min(100, battery_usage_percentage + (solar_surplus_prediction/battery_full_capacity*100) ))

            # Display results
            st.success("✅ Prediction ready")
            st.subheader("Prediction Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted solar energy surplus generated until sunset (KWh): ", f"{solar_surplus_prediction}", format="%.2f")
            col2.metric("Predicted battery capacity usage at sunset (including stored solar surplus) (%): ", f"{battery_predicted_usage}")
            col3.metric("Predicted solar energy surplus exported to the grid (KWh):", f"{solar_surplus_to_grid}",format="%.2f")
            
                    
    except Exception as e:
        st.error(f"⚠️ Exception: {e}")
    

if __name__=="__main__":
    main()




