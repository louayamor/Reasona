from Reasona.utils.logger import setup_logger
from typing import List, Dict
from pathlib import Path
import pandas as pd
import json

logger = setup_logger(__name__, "logs/data/formatter.json")


class DataFormatter:
    REQUIRED_COLUMNS = {"query", "synthetic_answer"}
    OPTIONAL_COLUMNS = {"synth_id", "model", "exercise", "script"}

    def __init__(self, df: pd.DataFrame):
        logger.info("Initializing DataFormatter")

        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = df.reset_index(drop=True)
        logger.info(f"Input dataframe shape: {self.df.shape}")

    def to_instruction_format(self) -> List[Dict]:
        logger.info("Formatting dataset to instruction format")

        formatted = []

        for idx, row in enumerate(self.df.itertuples(index=False)):
            try:
                item = {
                    "instruction": getattr(row, "query"),
                    "input": "",
                    "output": getattr(row, "synthetic_answer"),
                    "metadata": {
                        "id": getattr(row, "synth_id", None),
                        "model": getattr(row, "model", None),
                        "exercise": getattr(row, "exercise", None),
                        "script": getattr(row, "script", None),
                    },
                }

                formatted.append(item)

                if idx > 0 and idx % 1000 == 0:
                    logger.info(f"{idx} samples formatted")

            except Exception as e:
                logger.exception(f"Failed to format row {idx}: {e}")

        logger.info(f"Formatting completed: {len(formatted)} samples")
        return formatted

    def save_jsonl(self, dataset: List[Dict], path: Path) -> None:
        if not dataset:
            raise ValueError("Dataset is empty. Nothing to save.")

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving dataset to {path}")

        try:
            with open(path, "w", encoding="utf-8") as f:
                for idx, item in enumerate(dataset):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

                    if idx > 0 and idx % 5000 == 0:
                        logger.info(f"{idx} samples written")

            logger.info("JSONL dataset saved successfully")

        except Exception as e:
            logger.exception(f"Failed to save JSONL file: {e}")
            raise
