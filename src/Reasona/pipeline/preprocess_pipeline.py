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

        self.output_path = self.pre_cfg.merged_dir / "dataset_transformed.jsonl"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.processor = StreamingDatasetProcessor()

    def run(self):
        logger.info("=== PREPROCESSING PIPELINE STARTED ===")

        buffer = []
        count = 0

        for row in self.processor.stream_samples():
            buffer.append(row)
            count += 1

            if len(buffer) >= 1000:
                self._process_and_write(buffer)
                buffer = []

            if count % 5_000 == 0:
                logger.info(f"{count} samples processed")

            if self.pre_cfg.limit and count >= self.pre_cfg.limit:
                logger.info(f"Limit reached: {self.pre_cfg.limit}")
                break

        if buffer:
            self._process_and_write(buffer)

        logger.info(f"Preprocessing finished. Total samples: {count}")
        logger.info(f"Output saved to {self.output_path}")
        logger.info("=== PREPROCESSING PIPELINE FINISHED ===")

    def _process_and_write(self, rows):
        """
        Format streamed rows and append to JSONL.
        """
        df = pd.DataFrame(rows)
        formatter = DataFormatter(df)

        formatted = formatter.to_instruction_format()

        with open(self.output_path, "a", encoding="utf-8") as f:
            for item in formatted:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
