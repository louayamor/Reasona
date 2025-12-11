from Reasona.utils.logger import setup_logger
import pandas as pd
import json

# Unique logger per module
logger = setup_logger(__name__, "logs/data/formatter.log")


class DataFormatter:
    def __init__(self, df: pd.DataFrame):
        logger.info("Initializing DataFormatter")
        logger.info(f"Input dataframe shape: {df.shape}")
        self.df = df

    # ----------------------------------------
    # FORMAT INTO INSTRUCTION FORMAT
    # ----------------------------------------
    def to_instruction_format(self):
        logger.info("Starting to_instruction_format()")
        logger.info(f"Formatting {len(self.df)} rows")

        formatted = []

        for i, row in self.df.iterrows():
            try:
                item = {
                    "instruction": row["query"],
                    "input": "",
                    "output": row["synthetic_answer"],
                    "metadata": {
                        "id": row.get("synth_id"),
                        "model": row.get("model"),
                        "exercise": row.get("exercise"),
                        "script": row.get("script"),
                    },
                }
                formatted.append(item)

                if i > 0 and i % 1000 == 0:
                    logger.info(f"{i} rows formatted")

            except Exception as e:
                logger.exception(f"Error formatting row {i}: {e}")

        logger.info(f"Formatting completed. Total samples formatted: {len(formatted)}")
        return formatted

    # ----------------------------------------
    # SAVE TO JSONL
    # ----------------------------------------
    def save(self, dataset, file_path):
        logger.info(f"Saving {len(dataset)} items to {file_path}")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for i, item in enumerate(dataset):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    if i > 0 and i % 5000 == 0:
                        logger.info(f"{i} lines written")

            logger.info("Formatted dataset saved successfully")

        except Exception as e:
            logger.exception(f"Error saving file {file_path}: {e}")
