from pathlib import Path
from typing import Iterable
import numpy as np

from Reasona.data.embedder import Embedder
from Reasona.data.chunker import TextChunker
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.config.config_manager import IndexingConfig
from Reasona.utils.logger import setup_logger

logger = setup_logger("indexing_pipeline", "logs/pipeline/indexing_pipeline.json")


class IndexingPipeline:
    """
    Streaming-safe FAISS indexing pipeline.
    Never loads FAISS index.
    Uses metadata DB for capacity checks.
    """

    def __init__(self, cfg: IndexingConfig):
        self.cfg = cfg

        self.chunker = TextChunker(cfg.chunk_size, cfg.chunk_overlap)
        self.embedder = Embedder(
            model_name=cfg.embedding_model,
            device=cfg.device,
            batch_size=cfg.batch_size,
        )

        self.store = FaissStore(
            dim=cfg.embedding_dim,
            index_path=Path(cfg.vector_store_dir) / "index.faiss",
            db_path=Path(cfg.vector_store_dir) / "metadata.db",
            max_vectors=cfg.max_vectors,
        )

        self._stop = False

    def run(self, stream: Iterable[dict]):
        """
        Stream → chunk → embed → add to FAISS
        Stops automatically when max_vectors is reached.
        """

        # --- Capacity guard (NO FAISS LOAD) ---
        current_vectors = self.store.count_vectors()
        if current_vectors >= self.cfg.max_vectors:
            logger.warning(
                f"Max vectors reached ({current_vectors}/{self.cfg.max_vectors}). "
                "Indexing skipped."
            )
            return

        logger.info(f"Starting indexing from vector #{current_vectors}")

        for sample in stream:
            if self._stop:
                break

            texts = self.chunker.chunk(sample["text"])
            if not texts:
                continue

            embeddings = self.embedder.encode(texts)
            if embeddings is None or len(embeddings) == 0:
                continue

            remaining = self.cfg.max_vectors - self.store.count_vectors()
            if remaining <= 0:
                logger.warning("Max vectors reached during stream. Stopping.")
                break

            embeddings = embeddings[:remaining]
            texts = texts[:remaining]

            metadata = [
                {
                    "text": text,
                    "source": sample.get("source"),
                }
                for text in texts
            ]

            self.store.add(
                vectors=np.asarray(embeddings, dtype="float32"),
                metadatas=metadata,
            )

        self.store.save()
        logger.info("Indexing completed successfully.")
