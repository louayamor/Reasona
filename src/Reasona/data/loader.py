from datasets import load_dataset
from typing import Dict, Any, Iterator, Optional
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/loader.json")


class StreamingDatasetProcessor:
    def __init__(
        self,
        dataset_name: str,
        revision: str = "main",
        cache_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.revision = revision
        self.cache_dir = cache_dir

    def stream_samples(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        buffer_size: int = 10_000,
    ) -> Iterator[Dict[str, Any]]:
        logger.info(
            f"Starting streaming | dataset={self.dataset_name}, "
            f"split={split}, max_samples={max_samples}"
        )

        ds = load_dataset(
            self.dataset_name,
            split=split,
            streaming=True,
            revision=self.revision,
            cache_dir=self.cache_dir,
        )

        # Local shuffle only (HF limitation)
        if buffer_size and buffer_size > 0:
            ds = ds.shuffle(buffer_size=buffer_size)

        if max_samples is not None:
            ds = ds.take(max_samples)

        for sample in ds:
            yield sample
