from typing import Iterator, Dict, Any, Optional
import time
import threading

from Reasona.utils.logger import setup_logger
from Reasona.data.loader import StreamingDatasetProcessor
from Reasona.data.formatter import DataFormatter
from Reasona.entities.config_entity import PreprocessConfig

logger = setup_logger(__name__, "logs/pipeline/preprocess_pipeline.json")


class PreprocessPipeline:
    def __init__(self, cfg: PreprocessConfig):
        logger.info("Initializing PreprocessPipeline")
        self.cfg = cfg

        self.loader = StreamingDatasetProcessor(
            dataset_name=cfg.dataset_name,
            revision=cfg.revision,
            cache_dir=cfg.cache_dir,
        )

        self.formatter = DataFormatter()

        self._samples_processed = 0
        self._start_time: Optional[float] = None
        self._first_sample_time: Optional[float] = None
        self._lock = threading.Lock()

        if cfg.max_samples is None:
            logger.warning(
                "max_samples=None on a large streaming dataset. "
                "Ensure downstream stages do NOT materialize full data."
            )

    def stream(self) -> Iterator[Dict[str, Any]]:
        logger.info("=== PREPROCESS STREAM STARTED ===")

        self._start_time = time.time()
        self._first_sample_time = None
        self._samples_processed = 0

        stream = self.loader.stream_samples(
            split=self.cfg.split,
            max_samples=self.cfg.max_samples,
            buffer_size=self.cfg.buffer_size,
        )

        try:
            for idx, raw_sample in enumerate(stream, start=1):
                with self._lock:
                    self._samples_processed += 1
                    if self._first_sample_time is None:
                        self._first_sample_time = time.time()
                        logger.info(
                            f"First sample processed | "
                            f"startup_time={self._first_sample_time - self._start_time:.2f}s"
                        )

                processed = self.formatter.format_sample(raw_sample)
                processed["_metadata"] = {
                    "index": idx,
                    "timestamp": time.time(),
                    "samples_per_second": self._throughput(),
                }

                yield processed

                if idx % 50_000 == 0:
                    logger.info(
                        f"Progress | samples={idx}, "
                        f"rate={self._throughput():.1f} samples/sec"
                    )

        finally:
            self._log_final_stats()

    def _throughput(self) -> float:
        if not self._start_time or self._samples_processed == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return self._samples_processed / elapsed if elapsed > 0 else 0.0

    def _log_final_stats(self):
        elapsed = time.time() - self._start_time if self._start_time else 0
        logger.info(
            f"=== PREPROCESS STREAM ENDED === | "
            f"samples={self._samples_processed}, "
            f"time={elapsed:.2f}s, "
            f"rate={self._throughput():.1f} samples/sec"
        )
