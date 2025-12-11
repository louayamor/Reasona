from Reasona.utils.logger import setup_logger
import pandas as pd

logger = setup_logger("logs/data/cleaner.log")


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        logger.info("Initializing DataCleaner")
        logger.info(f"Input dataframe shape: {df.shape}")
        self.df = df

    # ----------------------------------------
    # CLEANING STEPS
    # ----------------------------------------
    def clean(self) -> pd.DataFrame:
        logger.info("Starting clean()")
        logger.info("Performing cleaning operations")

        try:
            df = self.df.copy()
            logger.info("Dataframe successfully copied")
        except Exception as e:
            logger.error(f"Error copying dataframe: {e}")
            return pd.DataFrame()

        # -------------------------
        # Remove duplicates
        # -------------------------
        try:
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            logger.info(f"Duplicate rows removed: {before - after}")
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")

        # -------------------------
        # Remove missing critical fields
        # -------------------------
        critical_cols = ["query", "synthetic_answer"]
        missing_before = df[critical_cols].isna().any(axis=1).sum()
        try:
            df = df.dropna(subset=critical_cols)
            missing_after = df[critical_cols].isna().any(axis=1).sum()
            logger.info(
                f"Rows with missing critical fields removed: {missing_before - missing_after}"
            )
        except Exception as e:
            logger.error(f"Error dropping missing critical fields: {e}")

        # -------------------------
        # Final summary
        # -------------------------
        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        return df

    # ----------------------------------------
    # SAVE CLEANED DATA
    # ----------------------------------------
    def save(self, df: pd.DataFrame, file_path):
        file_path = str(file_path)
        logger.info(f"Saving cleaned dataset to {file_path}")

        try:
            df.to_parquet(file_path, index=False)
            logger.info("Cleaned dataset saved successfully")
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")
