from typing import Dict, Any, Optional
import yaml
from pathlib import Path
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/formatter.json")


class DataFormatter:
    def __init__(self, schema_path: str = "config/schema.yaml"):
        self.schema_path = Path(schema_path)
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Dataset schema not found: {self.schema_path}")

        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)

        self.content_fields = [
            k for k, v in schema.get("columns", {}).items() if v.get("role") == "content"
        ]
        self.metadata_fields = [
            k for k, v in schema.get("columns", {}).items() if v.get("role") == "metadata"
        ]

    def format_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        text_parts = []
        for field in self.content_fields:
            val = sample.get(field)
            if isinstance(val, dict):
                text_parts.extend(str(v).strip() for v in val.values() if isinstance(v, str) and v.strip())
            elif isinstance(val, str) and val.strip():
                text_parts.append(val.strip())

        final_text = "\n\n".join(text_parts)

        if not final_text:
            logger.warning("Formatted text is empty | sample keys=%s", list(sample.keys()))
            return None

        metadata = {field: sample.get(field) for field in self.metadata_fields}

        return {
            "text": final_text,
            **metadata,
            "_metadata": sample, 
        }
