from dataclasses import dataclass
from pathlib import Path

# ---------------------------
# DATA INGESTION
# ---------------------------
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path


# ---------------------------
# DATA VALIDATION
# ---------------------------
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: str
    unzip_data_dir: Path
    all_schema: dict


# ---------------------------
# DATA CLEANING
# ---------------------------
@dataclass(frozen=True)
class DataCleaningConfig:
    root_dir: Path
    data_path: Path


# ---------------------------
# DATA TRANSFORMATION
# ---------------------------
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path


# ---------------------------
# DATA TRAINING
# ---------------------------
@dataclass(frozen=True)
class DataTrainingConfig:
    root_dir: Path
    data_path: Path


# ---------------------------
# MODEL TRAINER
# ---------------------------
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    n_estimators: int
    criterion: str
    min_samples_split: int
    target_column: str


# ---------------------------
# MODEL EVALUATION
# ---------------------------
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str
