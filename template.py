import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s:'
)

project_name = "Reasona"

logs_dirs = [
    "logs",
    "logs/pipeline",
    "logs/data",
    "logs/system"
]
for d in logs_dirs:
    os.makedirs(d, exist_ok=True)
    logging.info(f"Created log directory: {d}")

artifacts_dirs = [
    "artifacts",
    "artifacts/data",
    "artifacts/vectorstore",
    "artifacts/model",
]
for d in artifacts_dirs:
    os.makedirs(d, exist_ok=True)
    logging.info(f"Created artifacts directory: {d}")

list_of_files = [
    f"src/{project_name}/__init__.py",

    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/config_manager.py",
    f"src/{project_name}/config/params.yaml",

    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/loader.py",
    f"src/{project_name}/data/cleaner.py",
    f"src/{project_name}/data/formatter.py",
    f"src/{project_name}/data/chunker.py",
    f"src/{project_name}/data/embedder.py",

    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/preprocess_pipeline.py",
    f"src/{project_name}/pipeline/indexing_pipeline.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/inference_pipeline.py",

    f"src/{project_name}/vectorstore/__init__.py",
    f"src/{project_name}/vectorstore/faiss_store.py",

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/logger.py",
    f"src/{project_name}/utils/helpers.py",

    "config/config.yaml",
    "config/params.yaml",

    "main.py",

    "templates/index.html",

    "README.md",
]

for filepath in list_of_files:
    path = Path(filepath)
    dirpath = path.parent
    if dirpath and not dirpath.exists():
        os.makedirs(dirpath, exist_ok=True)
        logging.info(f"Ensured directory: {dirpath}")

    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w") as f:
            pass
        logging.info(f"Created empty file: {path}")
    else:
        logging.info(f"File exists: {path}")
