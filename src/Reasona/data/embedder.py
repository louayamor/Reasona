# Reasona/data/embedder.py
import torch
import numpy as np
from typing import Iterable, Iterator, List, Dict, Optional
from sentence_transformers import SentenceTransformer
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/embedder.json")


class Embedder:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)

        logger.info("Embedding model loaded on %s: %s", self.device, model_name)

    def embed_stream(
        self, items: Iterable[Dict]
    ) -> Iterator[tuple[np.ndarray, List[Dict]]]:
        """
        Streams items, batching them and returning embedded vectors with metadata.

        Expects each item to contain either:
        - "text" key (preferred)
        - "_metadata" key (optional)

        Skips items with empty or missing text.
        """
        buffer_texts: List[str] = []
        buffer_meta: List[Dict] = []
        total_items = 0
        total_batches = 0

        for idx, item in enumerate(items, start=1):
            # detect the text key
            text = item.get("text") or item.get("content") or ""
            if not text.strip():
                logger.warning("Skipping empty item at index %d | keys=%s", idx, list(item.keys()))
                continue

            buffer_texts.append(text)
            buffer_meta.append(item.get("_metadata", {}))
            total_items += 1

            if len(buffer_texts) >= self.batch_size:
                yield self._flush(buffer_texts, buffer_meta, total_batches + 1, total_items)
                buffer_texts, buffer_meta = [], []
                total_batches += 1

        # flush remaining items
        if buffer_texts:
            yield self._flush(buffer_texts, buffer_meta, total_batches + 1, total_items)

        logger.info("Embedder finished streaming | total_items=%d total_batches=%d", total_items, total_batches)

    def _flush(
        self, texts: List[str], metas: List[Dict], batch_idx: int, total_items: int
    ) -> tuple[np.ndarray, List[Dict]]:
        logger.info(
            "Embedding batch #%d | batch_size=%d | total_items=%d",
            batch_idx, len(texts), total_items
        )

        vectors = self.model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        logger.info(
            "Batch #%d embedded | vectors_shape=%s",
            batch_idx, vectors.shape
        )

        return vectors, metas
