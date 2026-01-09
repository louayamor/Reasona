import time
from pathlib import Path
from typing import Dict, Any, Iterable, Iterator, List
import gc

from Reasona.data.embedder import Embedder
from Reasona.data.chunker import TextChunker
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.config.config_manager import IndexingConfig
from Reasona.utils.logger import setup_logger


class IndexingPipeline:
    """
    Memory-optimized indexing pipeline for large datasets.
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
        self._max_reached_logged = False  # NEW

    def run(self, stream: Iterable[Dict[str, Any]]):
        chunk_stream = self._chunk_stream(stream)
        self._index_stream(chunk_stream)
        self._finalize()

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

    def _index_stream(self, chunk_stream: Iterable[Dict[str, Any]]):
        batch: List[Dict[str, Any]] = []

        for chunk in chunk_stream:
            if self.store and self.store.is_full:
                if not self._max_reached_logged:
                    self.logger.info(
                        "Max vectors reached (%d). Skipping further indexing.",
                        self.store.ntotal,
                    )
                    self._max_reached_logged = True
                continue

            batch.append(chunk)
            if len(batch) >= self.cfg.batch_size:
                self._process_batch(batch)
                batch.clear()
                gc.collect()

        if batch and not (self.store and self.store.is_full):
            self._process_batch(batch)
            batch.clear()
            gc.collect()

    def _process_batch(self, batch: List[Dict[str, Any]]):
        if self.store and self.store.is_full:
            return

        texts = [c["text"] for c in batch]
        metas = [
            {
                "id": c["id"],
                "text": c["text"],
                "source": c.get("source"),
                "original_id": c.get("original_id"),
            }
            for c in batch
        ]

        vectors = self.embedder.embed(texts)

        if self.store is None:
            self.store = FaissStore(
                dim=vectors.shape[1],
                index_path=self.index_path,
                db_path=self.meta_db_path,
                max_vectors=4_500_000,  
            )
            self.store.load()

        added = self.store.add(vectors, metas)
        self.vectors_written += added

        del vectors, metas, texts
        gc.collect()

        if added == 0:
            return

        if self.vectors_written % self.cfg.log_every < added:
            self._log_progress()

        if self.vectors_written % self.cfg.save_every < added:
            self._checkpoint()

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
            self.store._finalize()
            self.store.save()
            self.store._train_buffer.clear()
            gc.collect()

            self.logger.info(
                "Checkpoint saved | vectors=%d | RAM cleaned",
                self.vectors_written,
            )

    def _finalize(self):
        if self.store:
            self.store._finalize()
            self.store.save()
            self.store.close()
            gc.collect()

        self.logger.info(
            "Indexing finished | vectors=%d | runtime=%.1fs",
            self.vectors_written,
            time.time() - self.start_time,
        )
