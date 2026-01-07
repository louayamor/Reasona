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

        self.required_columns: Dict[str, type] = {
            col: self._map_dtype(meta.get("type", "string"))
            for col, meta in self.schema.get("columns", {}).items()
        }

        self.text_fields: List[str] = (
            self.schema.get("indexing", {}).get("text_fields", [])
        )

        self.metadata_fields: List[str] = (
            self.schema.get("indexing", {}).get("metadata_fields", [])
        )

        logger.info(
            "Validator initialized | required_columns=%s | text_fields=%s",
            list(self.required_columns.keys()),
            self.text_fields,
        )

    @staticmethod
    def _map_dtype(dtype: str) -> type:
        return {
            "string": str,
            "int": int,
            "float": float,
            "bool": bool,
        }.get(dtype, str)

    def is_valid(self, sample: Dict[str, Any]) -> bool:

        for col, expected_type in self.required_columns.items():
            if col not in sample:
                logger.warning("Missing column '%s' | keys=%s", col, list(sample.keys()))
                return False

            value = sample[col]

            if value is None:
                logger.warning("Null value for column '%s'", col)
                return False

            if not isinstance(value, expected_type):
                logger.warning(
                    "Wrong type for column '%s' | got=%s expected=%s",
                    col,
                    type(value),
                    expected_type,
                )
                return False

        for field in self.text_fields:
            text = sample.get(field)
            if not isinstance(text, str) or not text.strip():
                logger.debug(
                    "Skipping empty Wikipedia article | field=%s", field
                )
                return False

        return True
