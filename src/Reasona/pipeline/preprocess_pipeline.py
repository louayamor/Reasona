from Reasona.utils.logger import setup_logger
from Reasona.config.configuration import ConfigurationManager

from Reasona.data.loader import combine_parquet_files, save_combined_data
from Reasona.data.cleaner import DataCleaner
from Reasona.data.formatter import DataFormatter

logger = setup_logger("logs/pipeline/preprocess_pipeline.log")


class PreprocessPipeline:
    def __init__(self):
        logger.info("Initializing PreprocessPipeline()")

        cfg = ConfigurationManager()
        self.ingest_cfg = cfg.get_data_ingestion_config()
        self.clean_cfg = cfg.get_data_cleaning_config()
        self.transform_cfg = cfg.get_data_transformation_config()

    # --------------------------------
    # 1. INGESTION
    # --------------------------------
    def run_ingestion(self):
        logger.info("Running ingestion stage...")

        df = combine_parquet_files(limit=100_000)

        if df.empty:
            logger.error("Ingestion returned an empty DataFrame")
        else:
            logger.info(f"Ingestion loaded {len(df)} rows")

        save_combined_data(df)
        return df

    # --------------------------------
    # 2. CLEANING
    # --------------------------------
    def run_cleaning(self, df):
        logger.info("Running cleaning stage...")

        cleaner = DataCleaner(df)
        df_clean = cleaner.clean()

        clean_path = self.clean_cfg.data_path
        cleaner.save(df_clean, clean_path)

        logger.info(f"Cleaning completed. Cleaned rows: {len(df_clean)}")
        return df_clean

    # --------------------------------
    # 3. TRANSFORMATION
    # --------------------------------
    def run_transformation(self, df):
        logger.info("Running transformation stage...")

        formatter = DataFormatter(df)
        dataset = formatter.to_instruction_format()

        outfile = self.transform_cfg.data_path
        formatter.save(dataset, outfile)

        logger.info("Transformation completed.")
        return dataset

    # --------------------------------
    # MAIN PIPELINE
    # --------------------------------
    def run(self):
        logger.info("=== PREPROCESSING PIPELINE STARTED ===")

        df_ingested = self.run_ingestion()
        df_cleaned = self.run_cleaning(df_ingested)
        self.run_transformation(df_cleaned)

        logger.info("=== PREPROCESSING PIPELINE FINISHED ===")


if __name__ == "__main__":
    PreprocessPipeline().run()
