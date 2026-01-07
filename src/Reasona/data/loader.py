from datasets import load_dataset
from typing import Iterator, Dict, Any, Optional
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/loader.json")


class StreamingDatasetLoader:
    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.cache_dir = cache_dir

    def stream(
        self,
        split: str,
        shuffle_buffer: int = 0,
        max_samples: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:

        logger.info(
            "Loading dataset | name=%s config=%s split=%s streaming=True",
            self.dataset_name,
            self.dataset_config,
            split,
        )

        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=split,
            streaming=True,
            cache_dir=self.cache_dir,
        )

        if shuffle_buffer and shuffle_buffer > 0:
            logger.info("Shuffling stream | buffer_size=%d", shuffle_buffer)
            ds = ds.shuffle(buffer_size=shuffle_buffer)

        for i, sample in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                logger.info("Max samples reached | max_samples=%d", max_samples)
                break

            yield sample
