from Reasona.utils.logger import setup_logger
import pandas as pd
import json

logger = setup_logger("logs/data/formatter.log")


class DataFormatter:
    def __init__(self, df: pd.DataFrame):
        logger.info("Initializing DataFormatter")
        self.df = df

    # ----------------------------------------
    # FORMAT INTO INSTRUCTION FORMAT
    # ----------------------------------------
    def to_instruction_format(self):
        logger.info("Formatting dataset into instruction format")

        formatted = []

        for _, row in self.df.iterrows():
            item = {
                "instruction": row["query"],
                "input": "",
                "output": row["synthetic_answer"],
                "metadata": {
                    "id": row.get("synth_id", None),
                    "model": row.get("model", None),
                    "exercise": row.get("exercise", None),
                    "script": row.get("script", None),
                },
            }
            formatted.append(item)

        logger.info(f"Formatted {len(formatted)} samples")
        return formatted

    # ----------------------------------------
    # SAVE TO JSONL
    # ----------------------------------------
    def save(self, dataset, file_path):
        logger.info(f"Saving formatted dataset to {file_path}")

        with open(file_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info("Formatted dataset saved successfully")

