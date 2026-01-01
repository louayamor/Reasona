from datasets import load_dataset
from typing import Dict, Any, Iterator, Optional
import time

from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/loader.json")


class StreamingDatasetProcessor:
    """
    Hugging Face dataset processor with:
    - One-time metadata warmup (non-streaming)
    - True sample-level streaming
    """

    def __init__(
        self,
        dataset_name: str,
        revision: str = "main",
        cache_dir: str | None = None,
    ):
        if not isinstance(dataset_name, str):
            raise ValueError("dataset_name must be a string")

        self.dataset_name = dataset_name
        self.revision = revision
        self.cache_dir = cache_dir

        self._warmup_metadata()

    def _warmup_metadata(self) -> None:
        """
        Downloads dataset metadata once to reduce first-stream latency.
        """
        logger.info("Warming up dataset metadata (one-time)")
        load_dataset(
            self.dataset_name,
            split="train",
            streaming=False,
            revision=self.revision,
            cache_dir=self.cache_dir,
        )

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

        started = False
        start_time = time.time()

        for idx, sample in enumerate(dataset):
            if not started:
                logger.info("Streaming started")
                started = True

            yield sample

            if max_samples is not None and idx + 1 >= max_samples:
                break

        elapsed = time.time() - start_time
        logger.info(
            f"Streaming finished | samples={idx + 1 if started else 0}, time={elapsed:.1f}s"
        )
