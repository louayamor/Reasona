import time
from pathlib import Path
from typing import Dict, Any, Iterable, Iterator, List

import numpy as np

from Reasona.data.embedder import Embedder
from Reasona.data.chunker import TextChunker
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.config.config_manager import IndexingConfig
from Reasona.utils.logger import setup_logger


class IndexingPipeline:
    def __init__(self, cfg: IndexingConfig):
        self.cfg = cfg
        self.logger = setup_logger("indexing", "logs/pipeline/indexing.json")

        self.vector_db_dir = Path(cfg.vector_store_dir)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.vector_db_dir / "index.faiss"
        self.meta_path = self.vector_db_dir / "meta.pkl"

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

        if self.index_path.exists() and self.meta_path.exists():
            self.store = FaissStore.load(self.index_path, self.meta_path)
            self.vectors_written = self.store.ntotal
            self.logger.info(
                "Loaded existing FAISS index | vectors=%d", self.vectors_written
            )
        else:
            self.store = None
            self.vectors_written = 0

        self.start_time = time.time()

    def run(self, stream: Iterable[Dict[str, Any]]):
        
        chunk_stream = self._chunk_stream(stream)
        self._index_stream(chunk_stream)
        self._finalize()

    def _chunk_stream(self, stream: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        
        for item in stream:
            for chunk in self.chunker.chunk_item(item):
                yield {
                    "chunk_id": chunk["id"],
                    "text": chunk["text"],
                    "source": item.get("source"),
                }

    def _index_stream(self, chunk_stream: Iterable[Dict[str, Any]]):
        
        batch: List[Dict[str, Any]] = []

        for chunk in chunk_stream:
            batch.append(chunk)
            if len(batch) >= self.cfg.batch_size:
                self._process_batch(batch)
                batch.clear()

        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: List[Dict[str, Any]]):
        
        texts = [c["text"] for c in batch]
        metas = [{"chunk_id": c["chunk_id"], "source": c["source"]} for c in batch]

        vectors = self.embedder.embed(texts)

        if self.store is None:
            self.store = FaissStore(dim=vectors.shape[1])

        self.store.add(vectors, metas)
        self.vectors_written += len(vectors)

        if self.vectors_written % self.cfg.log_every < len(vectors):
            self._log_progress()

        if self.vectors_written % self.cfg.save_every < len(vectors):
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
        self.store.save(self.index_path, self.meta_path)
        self.logger.info("Checkpoint saved | vectors=%d", self.vectors_written)

    def _finalize(self):
        if self.store:
            self.store.save(self.index_path, self.meta_path)
        self.logger.info(
            "Indexing finished | vectors=%d | runtime=%.1fs",
            self.vectors_written,
            time.time() - self.start_time,
        )
