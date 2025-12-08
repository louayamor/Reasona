from Reasona.utils.logger import setup_logger
import pandas as pd

logger = setup_logger("logs/data/cleaner.log")


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        logger.info("Initializing DataCleaner")
        self.df = df

    # ----------------------------------------
    # CLEANING STEPS
    # ----------------------------------------
    def clean(self) -> pd.DataFrame:
        logger.info("Starting data cleaning")

        df = self.df.copy()

        # Remove duplicates
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        logger.info(f"Removed {before - after} duplicate rows")

        # Remove rows with missing critical fields
        critical_cols = ["query", "synthetic_answer"]
        df = df.dropna(subset=critical_cols)
        logger.info("Dropped rows with missing critical fields")

        logger.info(f"Cleaning complete. Final rows: {len(df)}")
        return df

    # ----------------------------------------
    # SAVE CLEANED DATA
    # ----------------------------------------
    def save(self, df: pd.DataFrame, file_path):
        file_path = str(file_path)
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved cleaned dataset to {file_path}")

