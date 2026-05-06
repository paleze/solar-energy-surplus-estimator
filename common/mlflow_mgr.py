import os
import logging
from mlflow import mlflow, MlflowClient

from pathlib import Path
import pickle


import utils


class MlflowMgr:

    """
    Mlflow manager class to handle Mlflow client and operations
    """

    client =None
    experiment_name=None
    model_name=None
    model_production_alias=None
    app_logger = utils.app_logger


    def __init__(self, mlflow_tracking_uri=None, experiment_name="solar-surplus-estimator-experiment", model_name="solar-surplus-estimator", model_production_alias="production"):

        """
        Initialize mlflow
        """

        self.mlflow_tracking_uri=mlflow_tracking_uri if not None else os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")
        self.experiment_name=experiment_name
        self.model_name=model_name
        self.model_production_alias=model_production_alias
    
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)


        self.client = self.get_client()

    def get_client(self):

        """
        Return an Mlflow client instance
        """

        if self.client is None:
            self.client = MlflowClient(self.mlflow_tracking_uri)

        return self.client
    

    def get_production_model_version(self):

        """
        Get the current production model version from Mlflow model registry using the specified model name and production alias.  
        Returns:

            The model version object if found, None otherwise.
        """

        return self.client.get_model_version_by_alias(self.model_name, self.model_production_alias)
    

    def get_production_model(self):
        """
        Load the current production model from Mlflow model registry using the specified model name and production alias.
        Returns:

            The loaded model object.
        """
    
        return mlflow.xgboost.load_model(f"models:/{self.model_name}@{self.model_production_alias}")

    

    def get_best_model(self):

        """
        Get the best model from Mlflow experiment based on the lowest rmse_test metric value.
        Returns: 
            best_run_id: The run id of the best model
            best_run_rmse: The rmse_test metric value of the best model
            best_model_id: The model id of the best model in Mlflow model registry"""

        runs = mlflow.search_runs(filter_string="status='FINISHED'")
        best_run = runs.loc[ runs['metrics.rmse_test'].idxmin() ]
        best_run_id = best_run['run_id']
        best_run_rmse = best_run['metrics.rmse_test']
        self.app_logger.info(f'best run_id: {best_run_id}, rmse: {best_run_rmse}')

        model_filter_string=f"source_run_id='{best_run_id}'"
        models = mlflow.search_logged_models(filter_string=model_filter_string)
        best_model_id=models.at[0,"model_id"]
        
        return best_run_id, best_run_rmse, best_model_id


    def register_best_model(self, dv):
            
        """
        Register the best model in Mlflow model registry and set it as production model using the specified model name and production alias.
        Args:
            dv: The data preprocesser DictionaryVectorizer to be saved as an artifact of the best model run.    
        """

        # Get best model
        best_run_id, best_run_rmse, best_model_id = self.get_best_model()


        # Get current registered model
        try:

            model_info = self.client.get_model_version_by_alias(self.model_name, self.model_production_alias)

        except mlflow.exceptions.RestException as e:
            # Model name or alias not found
            model_info = None

        if model_info is not None and model_info.run_id==best_run_id:
            
            best_version = model_info.version
            self.app_logger.info(f"Current model {self.model_name} with version {best_version} is still the best one")

        else:

            # register and set alias to a new model
            model_uri = f"runs:/{best_run_id}/model"
            best_version = mlflow.register_model(model_uri, self.model_name).version
        
            self.client.set_registered_model_alias(
                self.model_name, self.model_production_alias, best_version
            )

            # set temp local model path
            model_path=Path("/tmp/models")
            model_path.mkdir(parents=True, exist_ok=True)
            preprocessor_file=Path.joinpath(model_path,"preprocessor.b")

            # Save preprocessor DictionaryVectorzizer as a best run artifact
            with open(preprocessor_file, "wb") as f_out:
                pickle.dump(dv, f_out)

            mlflow.log_artifact(preprocessor_file, artifact_path="preprocessor", run_id=best_run_id)

            self.app_logger.info(f"Registered model '{self.model_name}' with new version '{best_version}' has now alias '{self.model_production_alias}'")

        
        return { 
                "best_run_id": best_run_id,
                "best_run_rmse": best_run_rmse,
                "best_model_id": best_model_id,
                "best_version": best_version
        }
    
