import utils
import requests
import jwt
import time
from datetime import datetime, timezone 


class AirflowMgr:

    airflow_access_token = None
    airflow_url = None
    app_logger = utils.app_logger
    
    def __init__(self, airflow_url="http://airflow-apiserver:8080", model_training_pipeline_dag_id="solar_energy_surplus_estimator_model_pipeline"):
        self.airflow_url = airflow_url
        self.model_training_pipeline_dag_id = model_training_pipeline_dag_id


    def get_access_token(self):
        """
        Get Airflow access token for authentication using Airflow credentials stored in docker secrets
        """

        # check if airflow access token is still valid
        if self.is_jwt_token_valid(self.airflow_access_token):
            return self.airflow_access_token

        self.app_logger.info("Requesting aiflow token")

        endpoint = f"{self.airflow_url}/auth/token"

        # Get Airflow credentials from docker secrets
        airflow_user=utils.get_secret_from_env_or_file(file_path_var="/run/secrets/airflow_user")
        airflow_password=utils.get_secret_from_env_or_file(file_path_var="/run/secrets/airflow_password")
        
        payload = {
            "username": f"{airflow_user}",
            "password": f"{airflow_password}",
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            
            token_data = response.json()
            self.airflow_access_token = token_data['access_token']

            self.app_logger.info(f"Aiflow token: {self.airflow_access_token}")

            return self.airflow_access_token
            
        except requests.exceptions.HTTPError as e:
            self.app_logger.error(f"HTTP error getting Airflow token: {e.response.status_code}")
            self.app_logger.error(f"Response: {e.response.text}")
            raise e
        except requests.exceptions.RequestException as e:
            self.app_logger.error(f"Network error getting Airflow token: {e}")
            raise e

        

    def is_jwt_token_valid(self, jwt_token):
        """
        Check if the given token is still valid by decoding it and checking the expiration time

        Args:
            token: JWT token to be checked
        """
        try:
            
            if jwt_token is None:
                return False

            decoded_token = jwt.decode(jwt_token, options={"verify_signature": False})
            exp_timestamp = decoded_token.get('exp')
            if exp_timestamp is None:
                raise RuntimeError("Token does not have an expiration time")
            
            current_timestamp = int(time.time())
            return current_timestamp < exp_timestamp
        
        except Exception as e:
            self.app_logger.error(f"Error checking token validity: {e}")
            return False


    def is_dag_running(self, dag_id=None):
        """
        Check if the dag_id has any active run in Airflow

        Args:
            dag_id: The DAG identifier
        """

        if dag_id is None:
            dag_id=self.model_training_pipeline_dag_id


        # Get access token
        access_token = self.get_access_token()

        endpoint = f"{self.airflow_url}/api/v2/dags/{dag_id}/dagRuns?order_by=-start_date"

        try:
            response = requests.get(
                endpoint,
                headers={
                    "Authorization": f"Bearer {access_token}"
                },
                timeout=10
            )
            response.raise_for_status()
            
            dag_runs = response.json()
            is_dag_running = False
            if 'dag_runs' in dag_runs and len(dag_runs['dag_runs']) > 0:
                latest_run = dag_runs['dag_runs'][0]
                is_dag_running = latest_run['state'] in ['running', 'queued']

            return is_dag_running
            
        except requests.exceptions.HTTPError as e:
            self.app_logger.error(f"HTTP error triggering DAG: {e.response.status_code}")
            self.app_logger.error(f"Response: {e.response.text}")
            raise e
        except Exception as e:
            self.app_logger.error(f"Unexpected error: {e}")
            raise e


    def trigger_airflow_dag(self, dag_id=None, conf=None, logical_date=None):
        """
        Trigger an Airflow DAG via REST API
        
        Args:
            dag_id: The DAG identifier
            conf: Dictionary of configuration parameters
            logical_date: ISO 8601 formatted execution date`
        
        Returns:
            DAG run ID if successful, None otherwise
        """

        if dag_id is None:
            dag_id=self.model_training_pipeline_dag_id

        if self.is_dag_running(dag_id):
            raise RuntimeError(f"DAG '{dag_id}' is already running! Please wait for the current run to finish before triggering a new one.")

        access_token = self.get_access_token()
        endpoint = f"{self.airflow_url}/api/v2/dags/{dag_id}/dagRuns"
        
        payload = {
            "logical_date": logical_date if logical_date else datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "note": "Triggered from model serving app after data drift detection"
        }

        
        if conf:
            payload["conf"] = conf
        
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()
            
            dag_run = response.json()
            dag_run_id = dag_run['dag_run_id']
            self.app_logger.info(f"DAG triggered successfully: {dag_run_id}")
            return dag_run_id
            
        except requests.exceptions.HTTPError as e:
            self.app_logger.error(f"HTTP error triggering DAG: {e.response.status_code}")
            self.app_logger.error(f"Response: {e.response.text}")
            raise e
        except requests.exceptions.RequestException as e:
            self.app_logger.error(f"Network error triggering DAG: {e}")
            raise e