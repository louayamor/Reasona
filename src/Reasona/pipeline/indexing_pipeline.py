import time
from pathlib import Path
from queue import Empty, Full
from multiprocessing import Process, Queue
from typing import Optional

from Reasona.data.embedder import Embedder
from Reasona.data.chunker import TextChunker
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.config.config_manager import IndexingConfig
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/pipeline/indexing_pipeline.json")


class IndexingPipelineMP:
    def __init__(self, cfg: IndexingConfig):
        self.cfg = cfg
        self.vector_db_dir = Path(cfg.vector_store_dir)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

        self.raw_queue: Queue = Queue(maxsize=cfg.queue_size)
        self.vec_queue: Queue = Queue(maxsize=cfg.queue_size * 3)

        self.embedder_process: Optional[Process] = None
        self.writer_process: Optional[Process] = None

        self.log_every = cfg.log_every
        self.save_every = cfg.save_every

        self.index_path = self.vector_db_dir / "index.faiss"
        self.meta_path = self.vector_db_dir / "meta.pkl"

    def start(self):
        logger.info("=== INDEXING PIPELINE STARTED ===")
        self.embedder_process = Process(target=self._embedder_worker, name="embedder-process")
        self.writer_process = Process(target=self._writer_worker, name="faiss-writer-process")

        self.embedder_process.start()
        self.writer_process.start()

    def index_chunks(self, item):
        while True:
            try:
                self.raw_queue.put(item, timeout=5)
                break
            except Full:
                logger.warning("Raw queue full, waiting to enqueue item")

    def stop(self):

        self.raw_queue.put(None)
        self.embedder_process.join()

        self.vec_queue.put(None)
        self.writer_process.join()
        logger.info("=== INDEXING PIPELINE FINISHED ===")

    # Embedder worker process

    def _embedder_worker(self):
        logger.info("Embedder process started")
        embedder = Embedder(self.cfg.embedding_model, batch_size=self.cfg.batch_size)
        chunker = TextChunker(self.cfg.chunk_size, self.cfg.chunk_overlap)

        items_seen = 0
        chunks_emitted = 0
        vectors_emitted = 0
        start_time = time.time()
        first_batch_logged = False

        def stream_chunks():
            nonlocal items_seen, chunks_emitted
            while True:
                try:
                    item = self.raw_queue.get(timeout=5)
                except Empty:
                    continue
                if item is None:
                    break
                items_seen += 1

                for chunk in chunker.chunk_item(item):
                    chunks_emitted += 1
                    chunk_text = chunk["text"]
                    chunk_meta = {
                        "text": chunk_text,
                        "source": item.get("source") if isinstance(item, dict) else None,
                        "original_text_length": len(item["text"].split()) if isinstance(item, dict) else len(item.split()),
                        "_metadata": item
                    }
                    yield chunk_meta

        for vectors, metas in embedder.embed_stream(stream_chunks()):
            while True:
                try:
                    self.vec_queue.put((vectors, metas), timeout=5)
                    break
                except Full:
                    logger.warning("Vector queue full, waiting to enqueue batch")
            vectors_emitted += vectors.shape[0]

            if not first_batch_logged:
                logger.info(
                    "First batch embedded | startup_time=%.2fs | batch_size=%d",
                    time.time() - start_time, vectors.shape[0]
                )
                first_batch_logged = True

            if vectors_emitted % self.log_every < vectors.shape[0]:
                elapsed = time.time() - start_time
                rate = vectors_emitted / max(elapsed, 1e-6)
                logger.info(
                    "Embedding progress | items=%d chunks=%d vectors=%d | avg_rate=%.1f vec/s",
                    items_seen, chunks_emitted, vectors_emitted, rate
                )

    # Writer worker process
    def _writer_worker(self):
        logger.info("FAISS writer process started")
        vectors_written = 0
        start_time = time.time()

        store: Optional[FaissStore] = None

        while True:
            try:
                item = self.vec_queue.get(timeout=5)
            except Empty:
                continue

            if item is None:
                break

            vectors, metas = item

            if store is None:
                dim = vectors.shape[1]
                store = FaissStore(dim=dim)
                logger.info("FAISS store initialized | dim=%d", dim)

            store.add(vectors, metas)
            vectors_written += vectors.shape[0]

            if vectors_written % self.log_every < vectors.shape[0]:
                rate = vectors_written / max(time.time() - start_time, 1e-6)
                logger.info(
                    "Indexing progress | vectors=%d | rate=%.1f vec/s",
                    vectors_written, rate
                )

            if vectors_written % self.save_every < vectors.shape[0]:
                store.save(self.index_path, self.meta_path)
                logger.info(
                    "Checkpoint saved | vectors=%d | path=%s",
                    vectors_written, self.vec_queue
                )

        if store is not None:
            store.save(self.index_path, self.meta_path)
            logger.info(
                "Final FAISS index saved | vectors=%d | runtime=%.1fs",
                vectors_written, time.time() - start_time
            )

        logger.info("FAISS writer process finished")
