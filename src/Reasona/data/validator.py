import yaml
from typing import Dict, Any, List
from pathlib import Path
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/pipeline/validator.json")


class Validator:

    def __init__(self, schema_path: str):
        schema_file = Path(schema_path)
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with schema_file.open("r", encoding="utf-8") as f:
            self.schema = yaml.safe_load(f)

        self.required_columns: Dict[str, type] = {}
        for col, meta in self.schema.get("columns", {}).items():
            dtype = meta.get("type", "string")

            if dtype == "string":
                self.required_columns[col] = str
            elif dtype == "int":
                self.required_columns[col] = int
            elif dtype == "float":
                self.required_columns[col] = float
            elif dtype == "bool":
                self.required_columns[col] = bool
            else:
                self.required_columns[col] = str  

        self.text_fields: List[str] = self.schema.get("indexing", {}).get("text_fields", [])
        self.metadata_fields: List[str] = self.schema.get("indexing", {}).get("metadata_fields", [])

    def validate(self, sample: Dict[str, Any]) -> bool:

        for col, col_type in self.required_columns.items():
            if col not in sample:
                logger.warning(f"Missing column '{col}' in sample: {list(sample.keys())}")
                return False
            if not isinstance(sample[col], col_type):
                logger.warning(
                    f"Column '{col}' has wrong type: {type(sample[col])}, expected {col_type}"
                )
                return False

        for field in self.text_fields:
            if not sample.get(field):
                logger.warning(f"Empty text field '{field}' in sample: {sample}")
                return False

        return True
