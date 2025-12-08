from Reasona.utils.logger import setup_logger
from pathlib import Path
import pandas as pd
import os

logger = setup_logger("logs/data/data_loader.log")

RAW_DIR = Path("artifacts/data_ingestion/raw")
COMBINED_DIR = Path("artifacts/data_ingestion/combined")
COMBINED_FILE = COMBINED_DIR / "combined_data.parquet"


def combine_parquet_files(limit=None):
    logger.info("Starting combine_parquet_files()")

    if not RAW_DIR.exists():
        logger.error(f"RAW_DIR does not exist: {RAW_DIR}")
        return pd.DataFrame()

    parquet_files = list(RAW_DIR.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files")

    if not parquet_files:
        return pd.DataFrame()

    combined_df = pd.DataFrame()
    loaded_rows = 0

    for idx, file_path in enumerate(parquet_files, start=1):
        logger.info(f"Loading {file_path}")

        try:
            df_chunk = pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue

        if limit:
            remaining = limit - loaded_rows
            df_chunk = df_chunk.head(remaining)

        combined_df = pd.concat([combined_df, df_chunk], ignore_index=True)
        loaded_rows = len(combined_df)

        if idx % 5 == 0 or loaded_rows % 100_000 == 0:
            logger.info(f"{loaded_rows} rows loaded so far")

        if limit and loaded_rows >= limit:
            break

    logger.info(f"Finished combining parquet files. Total rows: {len(combined_df)}")
    return combined_df


def save_combined_data(df):
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving combined dataset to {COMBINED_FILE}")
    df.to_parquet(COMBINED_FILE, index=False)
    logger.info(f"Saved combined dataset successfully")


if __name__ == "__main__":
    logger.info("loader.py started")
    df = combine_parquet_files(limit=100)
    if not df.empty:
        save_combined_data(df)
