
from datetime import datetime

import pandas as pd

import requests

import openmeteo_requests
import requests_cache
from retry_requests import retry

import logging



def get_weather_data(start_date, end_date, latitude, longitude, timezone):

    """
    Fetch archive or forecast weather data from Open-Meteo API according to date range, location and timezone
    """

    try:

        # Get archive or forecast data according to requested dates
        
        today = datetime.now().strftime("%Y-%m-%d")

        if start_date<today:

            logging.info(f"Getting historical weather data from {start_date} to {end_date}")
            
            assert(end_date >= start_date,f"end_date {end_date} must be equal or greater than start_date {start_date}")

            # Make sure all required weather variables are listed here
            # The order of variables in hourly or daily is important to assign them correctly below
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": {latitude},
                "longitude": {longitude},
                "start_date": {start_date},
                "end_date": { (end_date if end_date is not None else start_date) },
                "hourly": ["weather_code", "cloud_cover_high", "temperature"],
                "timezone": {timezone},
            }
        else:
            
            logging.info(f"Getting weather forecaast data for today {start_date}")

            url="https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": {latitude},
                "longitude": {longitude},
                "hourly": ["weather_code", "cloud_cover_high", "temperature"],
                "timezone": {timezone},
                "forecast_days": 1
            }


        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
 
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

        return hourly_df
    
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch historical weather data: {e}") from e
    
