import torch
import numpy as np
from typing import Iterable, Iterator, List, Dict, Optional
from sentence_transformers import SentenceTransformer
from Reasona.utils.logger import setup_logger
import time

logger = setup_logger(__name__, "logs/data/embedder.json")


class Embedder:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: Optional[str] = None,
        log_every: int = 50_000,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.log_every = log_every
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info("Embedding model loaded on %s: %s", self.device, model_name)

    def embed_stream(
        self, items: Iterable[Dict[str, any]]
    ) -> Iterator[tuple[np.ndarray, List[Dict[str, any]]]]:

        buffer_texts: List[str] = []
        buffer_meta: List[Dict] = []
        total_items = 0
        total_vectors = 0
        total_batches = 0
        start_time = time.time()
        first_batch_time: Optional[float] = None

        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise TypeError(f"Expected dict, got {type(item)} at index {idx}")

            text = item.get("text", "")
            metadata = item.get("_metadata", item)  

            if not text.strip():
                if idx % 10_000 == 0:
                    logger.warning("Skipping empty item at index %d", idx)
                continue

            buffer_texts.append(text)
            buffer_meta.append(metadata)
            total_items += 1

            if len(buffer_texts) >= self.batch_size:
                vectors, metas = self._flush(buffer_texts, buffer_meta)
                total_vectors += vectors.shape[0]
                total_batches += 1
                buffer_texts, buffer_meta = [], []

                if first_batch_time is None:
                    first_batch_time = time.time()
                    logger.info(
                        "First batch embedded | startup_time=%.2fs | batch_size=%d",
                        first_batch_time - start_time,
                        vectors.shape[0],
                    )

                if total_vectors % self.log_every < self.batch_size:
                    elapsed = time.time() - start_time
                    logger.info(
                        "Embedding progress | items=%d vectors=%d batches=%d | avg_rate=%.1f vec/s",
                        total_items, total_vectors, total_batches,
                        total_vectors / max(elapsed, 1e-6),
                    )

                yield vectors, metas

        if buffer_texts:
            vectors, metas = self._flush(buffer_texts, buffer_meta)
            total_vectors += vectors.shape[0]
            total_batches += 1
            yield vectors, metas

        elapsed_total = time.time() - start_time
        logger.info(
            "Embedding finished | total_items=%d total_vectors=%d total_batches=%d runtime=%.2fs | avg_rate=%.1f vec/s",
            total_items, total_vectors, total_batches, elapsed_total,
            total_vectors / max(elapsed_total, 1e-6),
        )

    def _flush(
        self, texts: List[str], metas: List[Dict]
    ) -> tuple[np.ndarray, List[Dict]]:
        vectors = self.model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        return vectors, metas

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        return vectors
