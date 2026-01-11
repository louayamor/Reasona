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
    Memory-safe indexing pipeline for FAISS with SQLite metadata.
    Supports max-vector cutoff and batch processing.
    """

    def __init__(self, cfg: IndexingConfig):
        self.cfg = cfg
        self.logger = setup_logger("indexing_pipeline", "logs/pipeline/indexing.json")

        # Prepare directories
        self.vector_dir = Path(cfg.vector_store_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.vector_dir / "index.faiss"
        self.meta_db_path = self.vector_dir / "metadata.db"

        # Embedding & chunker
        self.embedder = Embedder(
            model_name=cfg.embedding_model,
            batch_size=cfg.batch_size,
            device=getattr(cfg, "device", "cuda"),
            log_every=cfg.log_every,
        )
        self.chunker = TextChunker(
            chunk_size=cfg.chunk_size,
            overlap=cfg.chunk_overlap,
            log_every=cfg.log_every,
        )

        # FAISS store
        self.store: FaissStore | None = None
        self.vectors_written = 0
        self._buffer: List[Dict[str, Any]] = []
        self.start_time = time.time()

    # -------------------------
    # Public
    # -------------------------
    def run(self, stream: Iterable[Dict[str, Any]]):
        # Initialize FAISS store lazily
        self.store = FaissStore(
            dim=None,  # will be set on first batch
            index_path=self.index_path,
            db_path=self.meta_db_path,
            max_vectors=self.cfg.max_vectors,
            nprobe=getattr(self.cfg, "nprobe", 16),
        )
        self.store.load()

        self.vectors_written = self.store.count_vectors()
        if self.store.is_full:
            self.logger.warning(
                "Max vectors reached (%d/%d). Indexing skipped.",
                self.vectors_written,
                self.cfg.max_vectors,
            )
            return

        self.logger.info("Starting indexing from vector #%d", self.vectors_written)

        for chunk in self.chunker.chunk_stream(stream):
            if self.store.is_full:
                self.logger.warning(
                    "Max vectors reached (%d/%d). Indexing stopped.",
                    self.store.count_vectors(),
                    self.cfg.max_vectors,
                )
                break

            self._buffer.append(chunk)

            if len(self._buffer) >= self.cfg.batch_size:
                self._process_batch()

        # Process any remaining buffer
        if self._buffer:
            self._process_batch()

        self._finalize()

    # -------------------------
    # Batch processing
    # -------------------------
    def _process_batch(self):
        batch = self._buffer
        self._buffer = []

        texts = [c["text"] for c in batch]
        metas = batch

        vectors = self.embedder.embed(texts)

        # Initialize FAISS index on first batch
        if self.store.index is None:
            self.store._create_index(vectors.shape[1])

        added = self.store.add(vectors, metas)
        self.vectors_written += added

        del vectors, metas, texts, batch
        gc.collect()

        # Logging and checkpoint
        if self.vectors_written % self.cfg.log_every < self.cfg.batch_size:
            self._log_progress()
        if self.vectors_written % getattr(self.cfg, "save_every", 50_000) < self.cfg.batch_size:
            self._checkpoint()


    # -------------------------
    # Logging
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
            gc.collect()
            self.logger.info("Checkpoint saved | vectors=%d", self.vectors_written)

    # -------------------------
    # Finalize
    # -------------------------
    def _finalize(self):
        if self.store:
            self.store.finalize()
            self.store.close()
        self.logger.info(
            "Indexing finished | vectors=%d | runtime=%.1fs",
            self.vectors_written,
            time.time() - self.start_time,
        )
