from datasets import load_dataset
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
import json
import time
import signal

from Reasona.utils.retry_timeout import timeout_context, retry_on_exception
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/loader.jsonl")

DEFAULT_OUTPUT_FILE = Path("artifacts/data_ingestion/combined/streamed_synth.jsonl")
DEFAULT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


class StreamingDatasetProcessor:
    """
    Robust streaming processor for large Hugging Face datasets.
    Streams data sample-by-sample and writes to JSONL.
    """

    def __init__(
        self,
        dataset_name: str = "PleIAs/SYNTH",
        output_file: Optional[Path] = None,
        revision: str = "main",
        timeout_per_sample: int = 30,
        max_retries: int = 3,
        batch_size: int = 1000,
    ):
        self.dataset_name = dataset_name
        self.output_file = output_file or DEFAULT_OUTPUT_FILE
        self.revision = revision
        self.timeout_per_sample = timeout_per_sample
        self.max_retries = max_retries
        self.batch_size = batch_size
        self._stop_streaming = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Handle interrupts to stop streaming gracefully."""
        def handler(signum, frame):
            logger.warning("Interrupt received. Stopping streaming gracefully...")
            self._stop_streaming = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    @retry_on_exception(max_retries=3, delay=2.0)
    def _load_stream(self, split: str) -> Iterator[Dict[str, Any]]:
        """Load dataset stream with retry logic."""
        logger.info(f"Loading stream for {self.dataset_name} (split: {split})")
        dataset = load_dataset(
            self.dataset_name,
            split=split,
            streaming=True,
            revision=self.revision,
        )
        return iter(dataset)

    def stream_to_jsonl(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        batch_size: Optional[int] = None,  # can override instance batch_size
    ) -> Path:
        """Stream dataset to JSONL file."""
        batch_size = batch_size or self.batch_size
        logger.info(
            f"Streaming dataset={self.dataset_name}, split={split}, output={self.output_file}, batch_size={batch_size}"
        )

        try:
            dataset_iter = self._load_stream(split)
        except Exception as e:
            logger.error(f"Failed to load dataset stream: {e}")
            raise

        count = 0
        buffer = []
        consecutive_errors = 0
        start_time = time.time()

        with open(self.output_file, "w", encoding="utf-8") as f:
            while not self._stop_streaming:
                try:
                    with timeout_context(self.timeout_per_sample):
                        sample = next(dataset_iter)

                    consecutive_errors = 0
                    buffer.append(sample)

                    if len(buffer) >= batch_size:
                        self._write_batch(f, buffer)
                        count += len(buffer)
                        buffer.clear()
                        elapsed = time.time() - start_time
                        rate = count / elapsed if elapsed > 0 else 0
                        logger.info(f"{count} samples written (rate: {rate:.1f} samples/sec)")

                    if limit and count >= limit:
                        logger.info(f"Limit reached: {limit}")
                        break

                except StopIteration:
                    logger.info("End of dataset reached")
                    break
                except TimeoutError:
                    consecutive_errors += 1
                    logger.warning(f"Sample timeout (consecutive errors: {consecutive_errors})")
                    if consecutive_errors >= 5:
                        logger.error("Too many consecutive timeouts, stopping")
                        break
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error processing sample: {e}")
                    if consecutive_errors >= 10:
                        logger.error("Too many consecutive errors, stopping")
                        break
                    time.sleep(min(60, 2 ** consecutive_errors))

            # Flush remaining buffer
            if buffer:
                self._write_batch(f, buffer)
                count += len(buffer)

        total_time = time.time() - start_time
        logger.info(f"Streaming completed: {count} samples written in {total_time:.1f}s")
        return self.output_file

    @retry_on_exception(max_retries=3, delay=1.0)
    def stream_samples(
        self, split: str = "train", max_samples: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Yield samples one-by-one without writing to disk."""
        logger.info(f"Streaming samples from {self.dataset_name} ({split})")
        dataset_iter = self._load_stream(split)
        count = 0

        while not self._stop_streaming:
            if max_samples and count >= max_samples:
                break
            try:
                with timeout_context(self.timeout_per_sample):
                    sample = next(dataset_iter)
                yield sample
                count += 1
            except StopIteration:
                break
            except TimeoutError:
                logger.warning(f"Sample {count} timed out, skipping...")
            except Exception as e:
                logger.error(f"Error getting sample {count}: {e}")

    @staticmethod
    def _write_batch(file_handle, batch: list) -> None:
        """Write a batch of samples to disk safely."""
        try:
            for sample in batch:
                file_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            file_handle.flush()
        except Exception as e:
            logger.error(f"Error writing batch: {e}")
            raise


def stream_synth(
    split: str = "train",
    limit: Optional[int] = None,
    output_file: Optional[Path] = None,
    batch_size: int = 1000,
) -> Path:
    """Convenience wrapper for streaming SYNTH dataset."""
    processor = StreamingDatasetProcessor(
        output_file=output_file,
        timeout_per_sample=60,
        max_retries=5,
        batch_size=batch_size,
    )
    return processor.stream_to_jsonl(
        split=split,
        limit=limit,
        batch_size=batch_size,
    )
