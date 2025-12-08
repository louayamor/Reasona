# src/Reasona/config/config_manager.py

from pathlib import Path
import yaml
from Reasona.utils.helpers import read_yaml, create_directories
from Reasona.entities.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataCleaningConfig,
    DataTransformationConfig,
    DataTrainingConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

# Default config and params paths
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SCHEMA_FILE_PATH = Path("dataset_schema.yaml")


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Root artifacts directory
        create_directories([self.config.get("artifacts_root", "artifacts")])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        create_directories([cfg["root_dir"]])

        return DataIngestionConfig(
            root_dir=cfg["root_dir"],
            source_url=cfg["source_url"],
            local_data_file=cfg["local_data_file"]
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        cfg = self.config["data_validation"]
        create_directories([cfg["root_dir"]])

        return DataValidationConfig(
            root_dir=cfg["root_dir"],
            status_file=cfg["status_file"],
            unzip_data_dir=cfg["unzip_data_dir"],
            all_schema=self.schema["columns"]
        )

    def get_data_cleaning_config(self) -> DataCleaningConfig:
        cfg = self.config["data_cleaning"]
        create_directories([cfg["root_dir"]])

        return DataCleaningConfig(
            root_dir=cfg["root_dir"],
            data_path=cfg["data_path"]
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = self.config["data_transformation"]
        create_directories([cfg["root_dir"]])

        return DataTransformationConfig(
            root_dir=cfg["root_dir"],
            data_path=cfg["data_path"]
        )

    def get_data_training_config(self) -> DataTrainingConfig:
        cfg = self.config["data_training"]
        create_directories([cfg["root_dir"]])

        return DataTrainingConfig(
            root_dir=cfg["root_dir"],
            data_path=cfg["data_path"]
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self.config["model_trainer"]
        params = self.params.get("RandomForestClassifier", {})
        target_column = self.schema["target_column"]["name"]

        create_directories([cfg["root_dir"]])

        return ModelTrainerConfig(
            root_dir=cfg["root_dir"],
            train_data_path=cfg["train_data_path"],
            test_data_path=cfg["test_data_path"],
            model_name=cfg["model_name"],
            n_estimators=params.get("n_estimators", 100),
            criterion=params.get("criterion", "gini"),
            min_samples_split=params.get("min_samples_split", 2),
            target_column=target_column
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg = self.config["model_evaluation"]
        params = self.params.get("RandomForestClassifier", {})
        target_column = self.schema["target_column"]["name"]

        create_directories([cfg["root_dir"]])

        return ModelEvaluationConfig(
            root_dir=cfg["root_dir"],
            test_data_path=cfg["test_data_path"],
            model_path=cfg["model_path"],
            all_params=params,
            metric_file_name=cfg["metric_file_name"],
            target_column=target_column,
            mlflow_uri=cfg.get("mlflow_uri", "")
        )
