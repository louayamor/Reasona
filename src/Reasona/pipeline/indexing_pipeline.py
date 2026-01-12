import time
from pathlib import Path
from typing import Dict, Any, Iterable, List

from Reasona.data.embedder import Embedder
from Reasona.data.chunker import TextChunker
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.config.config_manager import IndexingConfig
from Reasona.utils.logger import setup_logger


class IndexingPipeline:
    """
    Robust FAISS indexing pipeline.
    - Resumable
    - Memory-safe
    - GPU-aware
    - Deterministic
    """

    def __init__(self, cfg: IndexingConfig):
        self.cfg = cfg
        self.logger = setup_logger(
            "indexing_pipeline",
            "logs/pipeline/indexing.json",
        )

        self.vector_dir = Path(cfg.vector_store_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.vector_dir / "index.faiss"
        self.db_path = self.vector_dir / "metadata.db"

        self.embedder = Embedder(
            model_name=cfg.embedding_model,
            batch_size=cfg.batch_size,
            device=cfg.device,
            log_every=cfg.log_every,
        )

        self.chunker = TextChunker(
            chunk_size=cfg.chunk_size,
            overlap=cfg.chunk_overlap,
            log_every=cfg.log_every,
        )

        self.store = FaissStore(
            dim=cfg.embedding_dim,
            index_path=self.index_path,
            db_path=self.db_path,
            max_vectors=cfg.max_vectors,
            nprobe=getattr(cfg, "nprobe", 16),
            mmap=False,
        )

        self.store.load()

        self.vectors_written = self.store.count_vectors()
        self.start_time = time.time()

        self.logger.info(
            "Indexing initialized | existing_vectors=%d | max_vectors=%s",
            self.vectors_written,
            cfg.max_vectors,
        )

    def run(self, stream: Iterable[Dict[str, Any]]):
        if self.store.is_full:
            self.logger.warning(
                "Index already full (%d/%d). Skipping indexing.",
                self.store.count_vectors(),
                self.cfg.max_vectors,
            )
            return

        buffer: List[Dict[str, Any]] = []

        for item in stream:
            for chunk in self.chunker.chunk_item(item):
                if self.store.is_full:
                    self.logger.warning(
                        "Max vectors reached (%d/%d). Stopping indexing.",
                        self.store.count_vectors(),
                        self.cfg.max_vectors,
                    )
                    self._finalize()
                    return

                buffer.append(chunk)

                if len(buffer) >= self.cfg.batch_size:
                    self._process_batch(buffer)
                    buffer.clear()

        if buffer:
            self._process_batch(buffer)

        self._finalize()

    def _process_batch(self, batch: List[Dict[str, Any]]):
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

        added = self.store.add(vectors, metas)
        self.vectors_written += added

        del vectors, metas, texts
        

        if self.vectors_written % self.cfg.log_every < self.cfg.batch_size:
            self._log_progress()

        if self.vectors_written % self.cfg.save_every < self.cfg.batch_size:
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
        self.store.save()
        
        self.logger.info(
            "Checkpoint saved | vectors=%d",
            self.vectors_written,
        )

    def _finalize(self):
        self.store.finalize()

        self.logger.info(
            "Indexing finished | vectors=%d | runtime=%.1fs",
            self.vectors_written,
            time.time() - self.start_time,
        )
