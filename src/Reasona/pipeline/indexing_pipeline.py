import time
import gc
from pathlib import Path
from typing import Dict, Any, Iterable, Iterator, List

from Reasona.data.embedder import Embedder
from Reasona.data.chunker import TextChunker
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.config.config_manager import IndexingConfig
from Reasona.utils.logger import setup_logger


class IndexingPipeline:
    """
    Memory-safe indexing pipeline with hard capacity limit.
    """

    def __init__(self, cfg: IndexingConfig):
        self.cfg = cfg
        self.logger = setup_logger("indexing", "logs/pipeline/indexing.json")

        self.vector_dir = Path(cfg.vector_store_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.vector_dir / "index.faiss"
        self.meta_db_path = self.vector_dir / "metadata.db"

        self.embedder = Embedder(
            model_name=cfg.embedding_model,
            batch_size=cfg.batch_size,
            log_every=cfg.log_every,
        )

        self.chunker = TextChunker(
            chunk_size=cfg.chunk_size,
            overlap=cfg.chunk_overlap,
            log_every=cfg.log_every,
        )

        self.store: FaissStore | None = None
        self.vectors_written = 0
        self.start_time = time.time()

    # -------------------------
    # Public API
    # -------------------------
    def run(self, stream: Iterable[Dict[str, Any]]):
        try:
            self._index_stream(self._chunk_stream(stream))
        finally:
            self._finalize()

    # -------------------------
    # Chunking
    # -------------------------
    def _chunk_stream(
        self, stream: Iterable[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        for item in stream:
            for chunk in self.chunker.chunk_item(item):
                yield {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "source": chunk.get("source"),
                    "original_id": item.get("id"),
                }

    # -------------------------
    # Indexing
    # -------------------------
    def _index_stream(self, chunk_stream: Iterable[Dict[str, Any]]):
        batch: List[Dict[str, Any]] = []

        for chunk in chunk_stream:
            batch.append(chunk)

            if len(batch) >= self.cfg.batch_size:
                if not self._process_batch(batch):
                    return
                batch.clear()
                gc.collect()

        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: List[Dict[str, Any]]) -> bool:
        texts = [c["text"] for c in batch]
        metas = batch

        vectors = self.embedder.embed(texts)

        if self.store is None:
            self.store = FaissStore(
                dim=vectors.shape[1],
                index_path=self.index_path,
                db_path=self.meta_db_path,
                max_vectors=self.cfg.max_vectors,
            )
            self.store.load()

        written = self.store.add(vectors, metas)
        self.vectors_written += written

        del vectors, texts, metas
        gc.collect()

        if written == 0 and self.store.is_full:
            self.logger.warning(
                "Vector store full (%d vectors). Indexing stopped.",
                self.store.ntotal,
            )
            return False

        if self.vectors_written % self.cfg.log_every < self.cfg.batch_size:
            self._log_progress()

        if self.vectors_written % self.cfg.save_every < self.cfg.batch_size:
            self._checkpoint()

        return True

    # -------------------------
    # Logging / Persistence
    # -------------------------
    def _log_progress(self):
        elapsed = max(time.time() - self.start_time, 1e-6)
        rate = self.vectors_written / elapsed
        self.logger.info(
            "Indexing progress | vectors=%d | rate=%.1f vec/s",
            self.vectors_written,
            rate,
        )

    def _checkpoint(self):
        if self.store:
            self.store.save()
            self.logger.info(
                "Checkpoint saved | vectors=%d", self.vectors_written
            )

    def _finalize(self):
        if self.store:
            self.store._finalize()
            self.store.save()
            self.store.close()

        self.logger.info(
            "Indexing finished | vectors=%d | runtime=%.1fs",
            self.vectors_written,
            time.time() - self.start_time,
        )
