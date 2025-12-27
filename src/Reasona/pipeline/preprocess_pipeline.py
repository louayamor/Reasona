from Reasona.utils.logger import setup_logger
from Reasona.config.config_manager import ConfigurationManager

from Reasona.data.loader import combine_parquet_files, save_combined_data
from Reasona.data.cleaner import DataCleaner
from Reasona.data.formatter import DataFormatter

logger = setup_logger(__name__, "logs/pipeline/preprocess_pipeline.json")


class PreprocessPipeline:
    def __init__(self):
        logger.info("Initializing PreprocessPipeline")
        cfg = ConfigurationManager()
        self.pre_cfg = cfg.get_preprocess_config()

    def run_ingestion(self):
        logger.info("Stage: Ingestion")

        df = combine_parquet_files(limit=self.pre_cfg.limit)
        if df is None or df.empty:
            raise RuntimeError("Ingestion failed: empty DataFrame")

        out_path = self.pre_cfg.combined_dir / "combined.parquet"
        save_combined_data(df, out_path)

        logger.info(f"Ingested {len(df)} rows")
        return df

    def run_cleaning(self, df):
        logger.info("Stage: Cleaning")

        cleaner = DataCleaner(df)
        df_clean = cleaner.clean()

        if df_clean.empty:
            raise RuntimeError("Cleaning failed: empty DataFrame")

        out_path = self.pre_cfg.processed_dir / "data_clean.csv"
        cleaner.save(df_clean, out_path)

        logger.info(f"Cleaned dataset saved to {out_path}")
        return df_clean

    def run_transformation(self, df):
        logger.info("Stage: Transformation → JSONL")

        formatter = DataFormatter(df)
        dataset = formatter.to_instruction_format()

        if not dataset:
            raise RuntimeError("Transformation failed: no samples produced")

        out_path = self.pre_cfg.merged_dir / "dataset_transformed.jsonl"
        formatter.save_jsonl(dataset, out_path)

        logger.info(f"Transformed dataset saved to {out_path}")
        return out_path

    def run(self):
        logger.info("=== PREPROCESSING PIPELINE STARTED ===")

        df = self.run_ingestion()
        df = self.run_cleaning(df)
        self.run_transformation(df)

        logger.info("=== PREPROCESSING PIPELINE FINISHED ===")


if __name__ == "__main__":
    PreprocessPipeline().run()
