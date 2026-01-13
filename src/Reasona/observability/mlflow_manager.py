import dagshub
import mlflow
from contextlib import contextmanager
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("mlflow_manager")
logger.setLevel(logging.INFO)


class MLflowManager:
    """
    MLflow manager integrated with Dagshub.
    Supports:
    - Automatic Dagshub initialization
    - Logging params, metrics, and artifacts
    - Optional automatic logging of model/generator config
    """

    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        experiment_name: str,
        auto_init: bool = True
    ):
        self.experiment_name = experiment_name

        if auto_init:
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            logger.info("Dagshub MLflow initialized | repo=%s/%s", repo_owner, repo_name)

        mlflow.set_experiment(experiment_name)
        logger.info("MLflowManager initialized | experiment=%s", experiment_name)

    @contextmanager
    def run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
        config: Optional[Any] = None
    ):
        try:
            with mlflow.start_run(run_name=run_name) as run:
                if tags:
                    mlflow.set_tags(tags)
                logger.info("MLflow run started | run_name=%s | run_id=%s", run_name, run.info.run_id)

                if config:
                    params = {k: v for k, v in vars(config).items() if not k.startswith("_")}
                    self.log_params(params)

                yield run
                logger.info("MLflow run completed | run_id=%s", run.info.run_id)
        except Exception as e:
            logger.error("MLflow run failed | run_name=%s | error=%s", run_name, str(e))
            raise

    @staticmethod
    def log_params(params: Dict[str, Any]):
        try:
            for k, v in params.items():
                mlflow.log_param(k, v)
            logger.info("MLflow params logged | keys=%s", list(params.keys()))
        except Exception as e:
            logger.error("Failed to log params | error=%s", str(e))

    @staticmethod
    def log_metrics(metrics: Dict[str, float]):
        try:
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            logger.info("MLflow metrics logged | keys=%s", list(metrics.keys()))
        except Exception as e:
            logger.error("Failed to log metrics | error=%s", str(e))

    @staticmethod
    def log_artifact(path: str, artifact_path: Optional[str] = None):
        try:
            mlflow.log_artifact(path, artifact_path=artifact_path)
            logger.info("MLflow artifact logged | path=%s | artifact_path=%s", path, artifact_path)
        except Exception as e:
            logger.error("Failed to log artifact | path=%s | error=%s", path, str(e))
