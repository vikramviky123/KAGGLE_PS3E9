import os
import sys
from pathlib import Path

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from concrete_strength.a_constants import *
from concrete_strength.b_entity.config_entity import ModelEvalConfig
from concrete_strength.c_config.configuration import ConfigurationManager
from concrete_strength.f_utils.common import save_pickle, load_pickle, read_yaml


class ModelEvaluation:
    def __init__(self, config: ModelEvalConfig):
        self.eval_config = config

    def read_params(self):
        params_yaml = read_yaml(Path(BESTPARAMS_FILE_PATH))
        return params_yaml

    def read_eval_results(self):
        loaded_results = load_pickle(self.eval_config.eval_results)
        return loaded_results

    def log_to_mlflow(self):
        params_yaml = self.read_params()
        loaded_results = self.read_eval_results()

        # remote_server_uri = "https://dagshub.com/vikramviky123/KAGGLE_PS3E9.mlflow"
        mlflow.set_registry_uri(self.eval_config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        for model_name, model_metrics in loaded_results.items():
            # Replace with your actual experiment name
            experiment_name = model_name + "_best_model"

            # Set the experiment name using mlflow.set_experiment
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():

                # Assuming 'test' metrics are used for evaluation, change as needed
                r2_sqr = np.mean(model_metrics['test']['r2_square'])
                mae = np.mean(model_metrics['test']['mae'])
                mse = np.mean(model_metrics['test']['mse'])
                r_mse = np.mean(model_metrics['test']['r_mse'])

                # Log best parameters to MLflow
                mlflow.log_params(params_yaml[model_name])

                # Log metrics to MLflow
                mlflow.log_metric("r2_sqr", r2_sqr)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r_mse", r_mse)

                # Set the experiment name as a tag (you can also use mlflow.set_tag)
                mlflow.set_tag("experiment_name", experiment_name)

                # Model registry does not work with file store
                # Set tracking_url_type_store based on your configuration
                tracking_url_type_store = "file"

                if tracking_url_type_store != "file":
                    # Register the model
                    mlflow.sklearn.log_model(
                        None, model_name, registered_model_name=f"{model_name}_model")
                else:
                    mlflow.sklearn.log_model(None, model_name)
