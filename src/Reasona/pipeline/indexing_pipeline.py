# Reasona/pipeline/indexing_pipeline.py
import time
import pickle
from pathlib import Path
from queue import Queue, Full, Empty
from threading import Thread
from typing import Optional
from Reasona.data.embedder import Embedder
from Reasona.data.chunker import TextChunker
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.vectorstore.faiss_store_manager import FaissStoreManager
from Reasona.config.config_manager import IndexingConfig
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/pipeline/indexing_pipeline.json")


class IndexingPipeline:
    def __init__(self, cfg: IndexingConfig):
        self.cfg = cfg

        self.vector_db_dir = Path(cfg.vector_store_dir)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

        self.store_manager = FaissStoreManager(str(self.vector_db_dir), keep_versions=cfg.keep_versions)

        self.raw_queue = Queue(maxsize=cfg.queue_size)
        self.vec_queue = Queue(maxsize=cfg.queue_size)

        self.embedder_thread: Optional[Thread] = None
        self.writer_thread: Optional[Thread] = None

        self.log_every = cfg.log_every or 50_000
        self.save_every = cfg.save_every or 100_000  

    def start(self):
        logger.info("=== INDEXING PIPELINE STARTED ===")
        self.embedder_thread = Thread(target=self._embedder_worker, name="embedder-thread")
        self.writer_thread = Thread(target=self._writer_worker, name="faiss-writer-thread")

        self.embedder_thread.start()
        self.writer_thread.start()

    def index_chunks(self, item):
        while True:
            try:
                self.raw_queue.put(item, timeout=5)
                break
            except Full:
                logger.warning("Raw queue full, waiting to enqueue item")

    def stop(self):
        self.raw_queue.put(None)
        self.embedder_thread.join()
        self.vec_queue.put(None)
        self.writer_thread.join()
        logger.info("=== INDEXING PIPELINE FINISHED ===")

    def _embedder_worker(self):
        logger.info("Embedder thread started")

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
                        "source": item.get("source") if isinstance(item, dict) else None,
                        "original_text_length": len(item["text"].split()) if isinstance(item, dict) else len(item.split()),
                        "_metadata": item  
                    }
                    yield {"text": chunk_text, "metadata": chunk_meta}

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

    def _writer_worker(self):
        logger.info("FAISS writer started")

        store: Optional[FaissStore] = None
        vectors_written = 0
        all_metadata = []  
        start_time = time.time()

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

                latest_index = self.store_manager.load_latest_index()
                if latest_index is not None:
                    store = FaissStore(dim=dim)
                    store.index = latest_index
                    store.metadata = []  
                    logger.info(
                        "Resuming FAISS store from latest index | dim=%d", dim
                    )
                else:
                    store = FaissStore(dim=dim)
                    logger.info("FAISS initialized | dim=%d", dim)

            store.add(vectors, metas)
            all_metadata.extend(metas)
            vectors_written += vectors.shape[0]

            if vectors_written % self.log_every < vectors.shape[0]:
                rate = vectors_written / max(time.time() - start_time, 1e-6)
                logger.info(
                    "Indexing progress | vectors=%d total=%d | rate=%.1f vec/s",
                    vectors_written, len(store.metadata), rate
                )

            if vectors_written % self.save_every < vectors.shape[0]:
                path = self.store_manager.save_index(store.index)
                
                meta_path = Path(str(path)).with_suffix(".pkl")
                with open(meta_path, "wb") as f:
                    pickle.dump(all_metadata, f)
                logger.info(
                    "Incremental save | path=%s | total_vectors=%d | metadata saved",
                    path, len(store.metadata)
                )

        if store is not None:
            path = self.store_manager.save_index(store.index)
            meta_path = Path(str(path)).with_suffix(".pkl")
            with open(meta_path, "wb") as f:
                pickle.dump(all_metadata, f)
            logger.info(
                "Vector store saved | path=%s total_vectors=%d | metadata saved | runtime=%.1fs",
                path, len(store.metadata), time.time() - start_time
            )

        logger.info("FAISS writer finished")
