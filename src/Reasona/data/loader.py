from datasets import load_dataset
from Reasona.utils.logger import setup_logger
from pathlib import Path
import json
from typing import Optional, Dict, Any, Iterator
import signal

logger = setup_logger(__name__, "logs/data/loader.jsonl")

DEFAULT_OUTPUT_FILE = Path("artifacts/data_ingestion/combined/streamed_synth.jsonl")
DEFAULT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


class StreamingDatasetProcessor:
    """
    Efficient streaming processor for large Hugging Face datasets.
    """

    def __init__(
        self,
        dataset_name: str = "PleIAs/SYNTH",
        output_file: Optional[Path] = None,
        revision: str = "main",
    ):
        self.dataset_name = dataset_name
        self.output_file = output_file or DEFAULT_OUTPUT_FILE
        self.revision = revision
        self._stop_streaming = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        def handler(signum, frame):
            logger.warning("Interrupt received. Stopping streaming gracefully...")
            self._stop_streaming = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _load_stream(self, split: str):
        return load_dataset(
            self.dataset_name,
            split=split,
            streaming=True,
            revision=self.revision,
        )

    def stream_to_jsonl(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        batch_size: int = 1000,
    ) -> Path:
        """
        Stream dataset to JSONL on disk.

        Args:
            split: Dataset split
            limit: Max number of samples
            batch_size: Flush size to disk

        Returns:
            Path to written JSONL file
        """
        logger.info(
            f"Streaming dataset={self.dataset_name}, split={split}, output={self.output_file}"
        )

        dataset = self._load_stream(split)

        count = 0
        buffer = []

        with open(self.output_file, "w", encoding="utf-8") as f:
            for sample in dataset:
                if self._stop_streaming:
                    logger.warning("Streaming stopped by user")
                    break

                buffer.append(sample)

                if len(buffer) >= batch_size:
                    self._write_batch(f, buffer)
                    count += len(buffer)
                    buffer.clear()

                    if count % 10_000 == 0:
                        logger.info(f"{count} samples written")

                if limit and count >= limit:
                    logger.info(f"Limit reached: {limit}")
                    break

            if buffer:
                self._write_batch(f, buffer)
                count += len(buffer)

        logger.info(f"Streaming completed: {count} samples written")
        return self.output_file

    def stream_samples(self, split: str = "train") -> Iterator[Dict[str, Any]]:
        """
        Yield samples one-by-one without writing to disk.
        """
        logger.info(f"Streaming samples from {self.dataset_name} ({split})")

        dataset = self._load_stream(split)

        for sample in dataset:
            if self._stop_streaming:
                break
            yield sample

    @staticmethod
    def _write_batch(file_handle, batch: list) -> None:
        for sample in batch:
            file_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def stream_synth(
    split: str = "train",
    limit: Optional[int] = None,
    output_file: Optional[Path] = None,
) -> Path:
    """
    Convenience wrapper for streaming SYNTH.
    """
    processor = StreamingDatasetProcessor(output_file=output_file)
    return processor.stream_to_jsonl(split=split, limit=limit)
