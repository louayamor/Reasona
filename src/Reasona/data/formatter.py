from pathlib import Path
from typing import List, Dict, Any

import json
import pandas as pd

from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/formatter.json")


class DataFormatter:
    REQUIRED_COLUMNS = {"query", "synthetic_answer"}
    OPTIONAL_COLUMNS = {"synth_id", "model", "exercise", "script"}

    def __init__(self, df: pd.DataFrame):
        self._validate_dataframe(df)
        self.df = df.reset_index(drop=True)
        logger.info(f"DataFormatter initialized | shape={self.df.shape}")

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")

        missing = DataFormatter.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def to_instruction_format(self) -> List[Dict[str, Any]]:
        logger.info("Converting dataframe to instruction format")
        formatted: List[Dict[str, Any]] = []

        columns = list(self.df.columns)
        col_index = {col: idx for idx, col in enumerate(columns)}

        for idx, row in enumerate(self.df.itertuples(index=False, name=None)):
            try:
                formatted.append(self._format_row(row, col_index))

                if idx > 0 and idx % 1_000 == 0:
                    logger.info(f"{idx} samples formatted")

            except Exception as e:
                logger.exception(f"Failed to format row {idx}: {e}")

        logger.info(f"Formatting completed | total_samples={len(formatted)}")
        return formatted

    def _format_row(
        self, row: tuple, col_index: Dict[str, int]
    ) -> Dict[str, Any]:
        return {
            "instruction": row[col_index["query"]],
            "input": "",
            "output": row[col_index["synthetic_answer"]],
            "metadata": self._extract_metadata(row, col_index),
        }

    def _extract_metadata(
        self, row: tuple, col_index: Dict[str, int]
    ) -> Dict[str, Any]:
        metadata = {}
        for col in self.OPTIONAL_COLUMNS:
            metadata[col] = (
                row[col_index[col]] if col in col_index else None
            )
        return metadata

    @staticmethod
    def save_jsonl(dataset: List[Dict[str, Any]], path: Path) -> None:
        if not dataset:
            raise ValueError("Dataset is empty. Nothing to save.")

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving JSONL dataset | path={path}")

        try:
            with path.open("w", encoding="utf-8") as f:
                for idx, item in enumerate(dataset):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

                    if idx > 0 and idx % 5_000 == 0:
                        logger.info(f"{idx} samples written")

            logger.info("JSONL dataset saved successfully")

        except Exception as e:
            logger.exception("Failed to save JSONL file")
            raise
