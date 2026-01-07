from typing import Iterator, Dict, Any
from pathlib import Path
import time

from Reasona.data.loader import StreamingDatasetLoader
from Reasona.data.formatter import DataFormatter
from Reasona.data.validator import Validator
from Reasona.entities.config_entity import PreprocessConfig
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/pipeline/preprocess_pipeline.json")


class PreprocessPipeline:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

        self.loader = StreamingDatasetLoader(
            dataset_name=cfg.dataset_name,
            cache_dir=str(cfg.cache_dir) if cfg.cache_dir else None,
        )

        schema_path = str(cfg.schema_path) if cfg.schema_path else "config/dataset_schema.yaml"

        self.validator = Validator(schema_path=schema_path)

        self.formatter = DataFormatter(schema_path=schema_path)

    def stream(self) -> Iterator[Dict[str, Any]]:
        logger.info(
            f"=== PREPROCESS STREAM STARTED | dataset={self.cfg.dataset_name}, "
            f"split={self.cfg.split}, max_samples={self.cfg.max_samples} ==="
        )
        start_time = time.time()
        first_sample_time = None
        samples_processed = 0

        for raw_sample in self.loader.stream(
            split=self.cfg.split,
            max_samples=self.cfg.max_samples,
            shuffle_buffer=self.cfg.shuffle_buffer,
        ):
            
            if not self.validator.is_valid(raw_sample):
                logger.warning("Invalid sample skipped | keys=%s", list(raw_sample.keys()))
                continue

            if self.cfg.language and raw_sample.get("language") != self.cfg.language:
                continue

            processed = self.formatter.format_sample(raw_sample)
            if not processed.get("text"):
                continue

            samples_processed += 1
            if first_sample_time is None:
                first_sample_time = time.time()
                logger.info(
                    f"First sample processed | time_to_first_sample={first_sample_time - start_time:.2f}s"
                )

            yield processed

            if samples_processed % 50_000 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Preprocessing progress | samples={samples_processed}, "
                    f"avg_rate={samples_processed/elapsed:.1f} samples/sec"
                )

        elapsed = time.time() - start_time
        logger.info(
            f"=== PREPROCESS STREAM ENDED | total_samples={samples_processed}, "
            f"total_time={elapsed:.2f}s, avg_rate={samples_processed/elapsed:.1f} samples/sec ==="
        )
