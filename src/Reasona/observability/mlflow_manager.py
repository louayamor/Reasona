import mlflow
from contextlib import contextmanager
from typing import Dict, Any, Optional
import os
import logging


class MLflowManager:
    """
    MLflow Manager for Reasona:
    - Supports Dagshub remote tracking with username/password
    - Context-managed runs with tags
    - Logging of params, metrics, and artifacts
    - Safe exception handling with FAILED run status
    """

    def __init__(self, experiment_name: str):
        self.logger = logging.getLogger("mlflow_manager")
        self.logger.setLevel(logging.INFO)

        # Remote tracking
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            self.logger.info("MLflow tracking URI set to %s", tracking_uri)
        else:
            self.logger.warning("MLFLOW_TRACKING_URI not set, using local mlruns/")

        if username and password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
            self.logger.info("MLflow auth configured with username %s", username)
        else:
            self.logger.warning("MLFLOW_TRACKING_USERNAME/PASSWORD not set")

        mlflow.set_experiment(experiment_name)
        self.logger.info("MLflow experiment set to '%s'", experiment_name)

    @contextmanager
    def run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ):
        """
        Context manager for MLflow run.
        Marks run as FAILED if exception occurs.
        """
        run = mlflow.start_run(run_name=run_name, nested=nested)
        try:
            if tags:
                mlflow.set_tags(tags)
            self.logger.info("MLflow run started: %s", run_name)
            yield run
            mlflow.end_run()
            self.logger.info("MLflow run finished: %s", run_name)
        except Exception as e:
            mlflow.end_run(status="FAILED")
            self.logger.error("MLflow run failed: %s | Error: %s", run_name, e)
            raise

    @staticmethod
    def log_params(params: Dict[str, Any]):
        """
        Logs a dictionary of parameters to MLflow.
        """
        for k, v in params.items():
            mlflow.log_param(k, v)

    @staticmethod
    def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
        """
        Logs metrics to MLflow. Optional step parameter supported.
        """
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

    @staticmethod
    def log_artifact(path: str, artifact_path: Optional[str] = None):
        """
        Logs a local file or folder as an artifact.
        """
        mlflow.log_artifact(path, artifact_path)

    @staticmethod
    def log_artifacts(paths: list[str], artifact_path: Optional[str] = None):
        """
        Logs multiple artifacts.
        """
        for path in paths:
            mlflow.log_artifact(path, artifact_path)
