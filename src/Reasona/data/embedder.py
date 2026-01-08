import time
from typing import Iterable, Iterator, List, Dict, Tuple, Optional

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/embedder.json")


class Embedder:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        log_every: int,
        device: Optional[str] = None,
    ):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.log_every = log_every

        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()

        logger.info(
            "Embedding model loaded | device=%s model=%s | batch_size=%d",
            self.device,
            model_name,
            self.batch_size,
        )

    def embed_stream(
        self, items: Iterable[Dict[str, str]]
    ) -> Iterator[Tuple[np.ndarray, List[Dict]]]:
        texts: List[str] = []
        metas: List[Dict] = []
        total_vectors = 0
        total_batches = 0
        start_time = time.time()
        first_batch = True

        for item in items:
            text = item.get("text", "").strip()
            if not text:
                continue

            texts.append(text)
            metas.append(self._strip_text(item))

            if len(texts) >= self.batch_size:
                vectors = self._encode(texts)
                total_vectors += len(vectors)
                total_batches += 1
                self._log_progress(start_time, total_vectors, total_batches, first_batch)
                first_batch = False

                yield vectors, metas
                texts.clear()
                metas.clear()

        if texts:
            vectors = self._encode(texts)
            total_vectors += len(vectors)
            total_batches += 1
            yield vectors, metas

        self._log_final(start_time, total_vectors, total_batches)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            vectors = self.model.encode(
                texts,
                batch_size=len(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return vectors.astype(np.float32, copy=False)

    @staticmethod
    def _strip_text(item: Dict) -> Dict:
        meta = dict(item)
        meta.pop("text", None)
        return meta

    def _log_progress(
        self, start_time: float, total_vectors: int, total_batches: int, first_batch: bool
    ):
        elapsed = time.time() - start_time
        if first_batch:
            logger.info(
                "First batch embedded | startup_time=%.2fs | batch_size=%d",
                elapsed,
                self.batch_size,
            )
        if total_vectors % self.log_every < self.batch_size:
            logger.info(
                "Embedding progress | vectors=%d batches=%d | avg_rate=%.1f vec/s",
                total_vectors,
                total_batches,
                total_vectors / max(elapsed, 1e-6),
            )

    def _log_final(self, start_time: float, total_vectors: int, total_batches: int):
        elapsed = time.time() - start_time
        logger.info(
            "Embedding finished | vectors=%d batches=%d runtime=%.1fs | avg_rate=%.1f vec/s",
            total_vectors,
            total_batches,
            elapsed,
            total_vectors / max(elapsed, 1e-6),
        )
