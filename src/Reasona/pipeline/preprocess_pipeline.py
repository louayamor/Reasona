from typing import Iterator, Dict, Any, Optional
import time
from Reasona.data.loader import StreamingDatasetLoader
from Reasona.data.formatter import DataFormatter
from Reasona.entities.config_entity import PreprocessConfig
from Reasona.data.validator import Validator
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/pipeline/preprocess_pipeline.json")


class PreprocessPipeline:
    def __init__(self, cfg: PreprocessConfig, schema_path: str):
        self.cfg = cfg
        self.loader = StreamingDatasetLoader(
            dataset_name=cfg.dataset_name,
            cache_dir=str(cfg.cache_dir) if cfg.cache_dir else None,
        )
        self.formatter = DataFormatter()

        self.validator = Validator(schema_path=schema_path)

        self._start_time: Optional[float] = None
        self._first_sample_time: Optional[float] = None
        self._samples_processed: int = 0
        self.progress_interval = getattr(cfg, "progress_interval", 50_000)

        if cfg.max_samples is None:
            logger.warning(
                "max_samples=None on large streaming dataset. "
                "Ensure downstream stages are streaming-safe."
            )

    def stream(self) -> Iterator[Dict[str, Any]]:
        logger.info(
            f"=== PREPROCESS PIPELINE STARTED === | dataset={self.cfg.dataset_name}, "
            f"split={self.cfg.split}, max_samples={self.cfg.max_samples}"
        )

        self._start_time = time.time()
        self._first_sample_time = None
        self._samples_processed = 0

        for idx, raw_sample in enumerate(
            self.loader.stream(
                split=self.cfg.split,
                max_samples=self.cfg.max_samples,
                shuffle_buffer=self.cfg.shuffle_buffer,
            ),
            start=1,
        ):
            if self.cfg.language and raw_sample.get("language") != self.cfg.language:
                continue

            if not self.validator.validate(raw_sample):
                continue

            processed = self.formatter.format_sample(raw_sample)
            self._samples_processed += 1

            if self._first_sample_time is None:
                self._first_sample_time = time.time()
                logger.info(
                    f"=== PREPROCESS STREAM STARTED === | "
                    f"time_to_first_sample={self._first_sample_time - self._start_time:.2f}s"
                )

            processed["_metadata"] = {
                "index": idx,
                "timestamp": time.time(),
                "samples_processed": self._samples_processed,
                "elapsed_sec": time.time() - self._start_time,
            }

            yield processed

            if idx % self.progress_interval == 0:
                logger.info(
                    f"Preprocessing progress | samples={idx}, "
                    f"rate={self._throughput():.1f} samples/sec"
                )

        self._log_final_stats()

    def _throughput(self) -> float:
        if not self._start_time or self._samples_processed == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return self._samples_processed / elapsed if elapsed > 0 else 0.0

    def _log_final_stats(self):
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        logger.info(
            f"=== PREPROCESS PIPELINE ENDED === | "
            f"total_samples={self._samples_processed}, "
            f"total_time={elapsed:.2f}s, "
            f"avg_rate={self._throughput():.1f} samples/sec"
        )
