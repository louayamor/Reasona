from datasets import load_dataset
from typing import Iterator, Dict, Any


class StreamingDatasetLoader:
    def __init__(self, dataset_name: str, cache_dir=None):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir

    def stream(
        self,
        split: str,
        shuffle_buffer: int,
        max_samples: int | None,
    ) -> Iterator[Dict[str, Any]]:
        ds = load_dataset(
            self.dataset_name,
            split=split,
            streaming=True,
            cache_dir=self.cache_dir,
        )

        if shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=shuffle_buffer)

        for i, sample in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            yield sample
