from Reasona.utils.logger import setup_logger
from Reasona.config.config_manager import ConfigurationManager

from Reasona.data.loader import combine_parquet_files, save_combined_data
from Reasona.data.cleaner import DataCleaner
from Reasona.data.formatter import DataFormatter

logger = setup_logger(__name__, "logs/pipeline/preprocess_pipeline.log")


class PreprocessPipeline:
    def __init__(self):
        logger.info("Initializing PreprocessPipeline()")

        cfg = ConfigurationManager()
        self.pre_cfg = cfg.get_preprocess_config()

    def run_ingestion(self):
        logger.info("Running ingestion stage...")
        df = combine_parquet_files(limit=self.pre_cfg.limit)

        if df is None or df.empty:
            logger.error("Ingestion returned an empty DataFrame")
            return df

        logger.info(f"Ingestion loaded {len(df)} rows")

        out_path = self.pre_cfg.combined_dir / "combined.parquet"
        save_combined_data(df, out_path)

        logger.info(f"Combined dataset saved to {out_path}")
        return df

    def run_cleaning(self, df):
        logger.info("Running cleaning stage...")

        cleaner = DataCleaner(df)
        df_clean = cleaner.clean()

        out_path = self.pre_cfg.processed_dir / "data_clean.csv"
        cleaner.save(df_clean, out_path)

        logger.info(f"Cleaning completed. Cleaned rows: {len(df_clean)}")
        logger.info(f"Clean dataset saved to {out_path}")

        return df_clean

    def run_transformation(self, df):
        logger.info("Running transformation stage...")

        formatter = DataFormatter(df)
        dataset = formatter.to_instruction_format()

        out_path = self.pre_cfg.merged_dir / "dataset_transformed.csv"
        formatter.save(dataset, out_path)

        logger.info(f"Transformation completed. Output saved to {out_path}")
        return dataset

    def run(self):
        logger.info("=== PREPROCESSING PIPELINE STARTED ===")

        df_ingested = self.run_ingestion()
        if df_ingested is None or df_ingested.empty:
            logger.error("Stopping pipeline: ingestion returned no data.")
            return

        df_cleaned = self.run_cleaning(df_ingested)
        self.run_transformation(df_cleaned)

        logger.info("=== PREPROCESSING PIPELINE FINISHED ===")


if __name__ == "__main__":
    PreprocessPipeline().run()
