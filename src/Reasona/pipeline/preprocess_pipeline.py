from pathlib import Path
import json
import pandas as pd

from Reasona.utils.logger import setup_logger
from Reasona.config.config_manager import ConfigurationManager
from Reasona.data.loader import StreamingDatasetProcessor
from Reasona.data.formatter import DataFormatter

logger = setup_logger(__name__, "logs/pipeline/preprocess_pipeline.json")


class PreprocessPipeline:
    def __init__(self):
        logger.info("Initializing PreprocessPipeline")

        cfg = ConfigurationManager()
        self.pre_cfg = cfg.get_preprocess_config()

        # Ensure output directories exist
        self.pre_cfg.merged_dir.mkdir(parents=True, exist_ok=True)

        # Initialize robust streaming processor
        self.processor = StreamingDatasetProcessor(
            dataset_name="PleIAs/SYNTH",
            output_file=self.pre_cfg.merged_dir / "dataset_transformed.jsonl",
            timeout_per_sample=getattr(self.pre_cfg, "timeout_per_sample", 30),
            max_retries=getattr(self.pre_cfg, "max_retries", 3),
        )

    def run(self):
        logger.info("=== PREPROCESSING PIPELINE STARTED ===")

        # Stream dataset to JSONL with batching, retries, and timeout
        output_file = self.processor.stream_to_jsonl(
            split="train",
            limit=self.pre_cfg.limit,
            batch_size=1000
        )

        logger.info(f"Initial streaming complete. File: {output_file}")

        # Post-processing: apply DataFormatter for instruction format
        df = pd.read_json(output_file, lines=True)
        formatter = DataFormatter(df)
        formatted = formatter.to_instruction_format()

        # Overwrite JSONL with formatted data
        with open(output_file, "w", encoding="utf-8") as f:
            for item in formatted:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Preprocessing finished. Total samples: {len(df)}")
        logger.info(f"Output saved to {output_file}")
        logger.info("=== PREPROCESSING PIPELINE FINISHED ===")
