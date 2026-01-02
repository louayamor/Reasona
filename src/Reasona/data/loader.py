from datasets import load_dataset
from typing import Dict, Any, Iterator, Optional
import time

from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/loader.json")


class StreamingDatasetProcessor:
    """
    Streaming dataset loader from Hugging Face.
    Supports optional max_samples for testing and cache_dir for local caching.
    """

    def __init__(
        self,
        dataset_name: str,
        revision: str = "main",
        cache_dir: Optional[str] = None,
    ):
        if not isinstance(dataset_name, str):
            raise ValueError("dataset_name must be a string")

        self.dataset_name = dataset_name
        self.revision = revision
        self.cache_dir = cache_dir



    def stream_samples(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        logger.info(f"Starting stream | dataset={self.dataset_name}, split={split}")
        dataset = load_dataset(
            self.dataset_name,
            split=split,
            streaming=True,
            revision=self.revision,
            cache_dir=self.cache_dir,
        )

        start_time = time.time()
        for idx, sample in enumerate(dataset, start=1):
            if idx == 1:
                logger.info("Streaming first sample")

            yield sample

            if max_samples is not None and idx >= max_samples:
                break

        elapsed = time.time() - start_time
        logger.info(
            f"Streaming finished | samples={idx if 'idx' in locals() else 0}, "
            f"time={elapsed:.2f}s"
        )
