import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "Reasona"

# Logs folder structure ---------------------------------------------
logs_dirs = [
    "logs",
    "logs/training",
    "logs/data",
    "logs/inference",
    "logs/system"
]

for log_dir in logs_dirs:
    os.makedirs(log_dir, exist_ok=True)
    logging.info(f"Created log directory: {log_dir}")

# Project file structure ----------------------------------------------
list_of_files = [
    # Core project structure
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/loader.py",
    f"src/{project_name}/data/cleaner.py",
    f"src/{project_name}/data/formatter.py",

    f"src/{project_name}/training/__init__.py",
    f"src/{project_name}/training/train_lora.py",
    f"src/{project_name}/training/config.py",

    f"src/{project_name}/model/__init__.py",
    f"src/{project_name}/model/base_loader.py",
    f"src/{project_name}/model/merge.py",
    f"src/{project_name}/model/export.py",

    f"src/{project_name}/inference/__init__.py",
    f"src/{project_name}/inference/local_server.py",
    f"src/{project_name}/inference/generate.py",

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/helpers.py",
    f"src/{project_name}/utils/logger.py",

    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/preprocess_pipeline.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/inference_pipeline.py",

    # Configs
    "config/config.yaml",
    "params.yaml",
    "dataset_schema.yaml",

    # Main application entrypoints
    "main.py",
    "app.py",

    # Deployment / Packaging
    "Dockerfile",
    "requirements.txt",
    "setup.py",

    # Research & Jupyter
    "research/experiments.ipynb",
    "research/dataset_preview.ipynb",

    # Web UI templates
    "templates/index.html",

    # Tests
    "tests/test_data_pipeline.py",
    "tests/test_training.py",
    "tests/test_inference.py",

    # CLI
    "cli.py",

    # Documentation
    "README.md",
]

# File creation loop ----------------------------------------------------
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filename}")
