from datasets import load_dataset
from typing import Dict, Any, Iterator, Optional
import time

from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/loader.json")


class StreamingDatasetProcessor:
    """
    Hugging Face streaming-only processor.
    - No disk writes
    - Sample-level streaming
    - Safe for inference / retriever ingestion
    """

    def __init__(
        self,
        dataset_name: str = "PleIAs/SYNTH",
        revision: str = "main",
    ):
        self.dataset_name = dataset_name
        self.revision = revision

    def _load_stream(self, split: str):
        logger.info(f"Starting stream | dataset={self.dataset_name}, split={split}")
        return load_dataset(
            self.dataset_name,
            split=split,
            streaming=True,
            revision=self.revision,
        )

    # ------------------------------------------------------------------
    # SAMPLE STREAM (MiniGPT / inference / retriever)
    # ------------------------------------------------------------------
    def stream_samples(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        dataset = self._load_stream(split)

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
