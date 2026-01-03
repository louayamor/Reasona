import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional

from Reasona.config.config_manager import IndexingConfig
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/pipeline/indexing_pipeline.json")


class IndexingPipeline:
    def __init__(self, cfg: IndexingConfig):
        self.cfg = cfg

        self.vector_db_dir = Path(cfg.vector_store_dir)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

        self.raw_queue = Queue(maxsize=cfg.queue_size)
        self.vec_queue = Queue(maxsize=cfg.queue_size)

        self.embedder_thread: Optional[Thread] = None
        self.writer_thread: Optional[Thread] = None

        self.log_every = 50_000

    def start(self):
        logger.info("=== INDEXING PIPELINE STARTED ===")
        self.embedder_thread = Thread(target=self._embedder_worker, name="embedder-thread")
        self.writer_thread = Thread(target=self._writer_worker, name="faiss-writer-thread")

        self.embedder_thread.start()
        self.writer_thread.start()

    def index_chunks(self, item):
        self.raw_queue.put(item)

    def stop(self):
        self.raw_queue.put(None)
        self.embedder_thread.join()
        self.writer_thread.join()
        logger.info("=== INDEXING PIPELINE FINISHED ===")

    # ------------------ Workers ------------------
    def _embedder_worker(self):
        from Reasona.data.embedder import Embedder
        from Reasona.data.chunker import TextChunker

        logger.info("Embedder thread started")

        embedder = Embedder(self.cfg.embedding_model, batch_size=self.cfg.batch_size)
        chunker = TextChunker(self.cfg.chunk_size, self.cfg.chunk_overlap)

        items_seen = 0
        chunks_emitted = 0
        vectors_emitted = 0
        start_time = time.time()

        def stream():
            nonlocal items_seen, chunks_emitted
            while True:
                item = self.raw_queue.get()
                if item is None:
                    break
                items_seen += 1
                chunks = list(chunker.chunk_item(item))
                chunks_emitted += len(chunks)
                yield from chunks

        for vectors, metas in embedder.embed_stream(stream()):
            self.vec_queue.put((vectors, metas))
            vectors_emitted += vectors.shape[0]

            if vectors_emitted % self.log_every == 0:
                now = time.time()
                rate = vectors_emitted / max(now - start_time, 1e-6)
                logger.info(
                    "Embedding progress | items=%d chunks=%d vectors=%d avg_rate=%.1f vec/s",
                    items_seen, chunks_emitted, vectors_emitted, rate
                )

        self.vec_queue.put(None)
        logger.info(
            "Embedder finished | items=%d chunks=%d vectors=%d runtime=%.1fs",
            items_seen, chunks_emitted, vectors_emitted, time.time() - start_time
        )

    def _writer_worker(self):
        from Reasona.vectorstore.faiss_store import FaissStore

        logger.info("FAISS writer started")

        store: Optional[FaissStore] = None
        vectors_written = 0
        start_time = time.time()

        while True:
            item = self.vec_queue.get()
            if item is None:
                break

            vectors, metas = item

            if store is None:
                store = FaissStore(dim=vectors.shape[1])
                logger.info("FAISS initialized | dim=%d", vectors.shape[1])

            store.add(vectors, metas)
            vectors_written += vectors.shape[0]

            if vectors_written % self.log_every == 0:
                rate = vectors_written / max(time.time() - start_time, 1e-6)
                logger.info(
                    "Indexing progress | vectors=%d total=%d rate=%.1f vec/s",
                    vectors_written, store.ntotal, rate
                )

        if store is not None:
            store.save(self.vector_db_dir)
            logger.info(
                "Vector store saved | path=%s total_vectors=%d runtime=%.1fs",
                self.vector_db_dir, store.ntotal, time.time() - start_time
            )

        logger.info("FAISS writer finished")
