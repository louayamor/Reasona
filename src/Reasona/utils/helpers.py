import yaml
from pathlib import Path
import os

def read_yaml(path: Path) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_directories(paths: list):
    """
    Create all directories in the provided list.
    """
    for path in paths:
        path = Path(path)
        os.makedirs(path, exist_ok=True)
