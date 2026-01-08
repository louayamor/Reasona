import time
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Thread
from typing import Dict, Any, Optional

from Reasona.data.embedder import Embedder
from Reasona.data.chunker import TextChunker
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.config.config_manager import IndexingConfig
from Reasona.utils.logger import setup_logger


class IndexingPipeline:
    def __init__(self, cfg: IndexingConfig):
        self.cfg = cfg
        self.vector_db_dir = Path(cfg.vector_store_dir)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

        self.raw_queue: Queue = Queue(maxsize=cfg.queue_size)
        self.vec_queue: Queue = Queue(maxsize=cfg.queue_size * 3)

        self.embedder_thread: Optional[Thread] = None
        self.writer_thread: Optional[Thread] = None

        self.index_path = self.vector_db_dir / "index.faiss"
        self.meta_path = self.vector_db_dir / "meta.pkl"

    def start(self):
        """Start embedder and writer threads"""
        self.embedder_thread = Thread(
            target=self._embedder_worker, 
            args=(self.raw_queue, self.vec_queue, self.cfg),
            name="embedder-thread"
        )
        self.writer_thread = Thread(
            target=self._writer_worker, 
            args=(self.vec_queue, self.index_path, self.meta_path, self.cfg),
            name="faiss-writer-thread"
        )

        self.embedder_thread.start()
        self.writer_thread.start()

    def index_chunks(self, item: Dict[str, Any] | str):
        while True:
            try:
                self.raw_queue.put(item, timeout=5)
                break
            except Full:
                print("Raw queue full, waiting to enqueue item...")

    def stop(self):
        self.raw_queue.put(None)
        self.embedder_thread.join()

        self.vec_queue.put(None)
        self.writer_thread.join()
        print("=== INDEXING PIPELINE FINISHED ===")

    @staticmethod
    def _embedder_worker(raw_queue: Queue, vec_queue: Queue, cfg: IndexingConfig):
        logger = setup_logger("embedder", "logs/pipeline/embedder.json")
        logger.info("Embedder thread started")

        embedder = Embedder(cfg.embedding_model, batch_size=cfg.batch_size)
        chunker = TextChunker(cfg.chunk_size, cfg.chunk_overlap)

        items_seen = 0
        chunks_emitted = 0
        vectors_emitted = 0
        start_time = time.time()
        first_batch_logged = False

        def stream_chunks():
            nonlocal items_seen, chunks_emitted
            while True:
                try:
                    item = raw_queue.get(timeout=5)
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
                    vec_queue.put((vectors, metas), timeout=5)
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

            if vectors_emitted % cfg.log_every < vectors.shape[0]:
                elapsed = time.time() - start_time
                rate = vectors_emitted / max(elapsed, 1e-6)
                logger.info(
                    "Embedding progress | items=%d chunks=%d vectors=%d | avg_rate=%.1f vec/s",
                    items_seen, chunks_emitted, vectors_emitted, rate
                )

        logger.info("Embedder thread finished")


    @staticmethod
    def _writer_worker(vec_queue: Queue, index_path: Path, meta_path: Path, cfg: IndexingConfig):
        logger = setup_logger("writer", "logs/pipeline/faiss_writer.json")
        logger.info("FAISS writer thread started")

        store: FaissStore
        if index_path.exists() and meta_path.exists():
            store = FaissStore.load(index_path, meta_path)
            processed_ids = set(store.metadata)  
            vectors_written = store.ntotal
            logger.info("Loaded existing FAISS index | vectors=%d", vectors_written)
        else:
            store = None
            processed_ids = set()
            vectors_written = 0

        start_time = time.time()

        while True:
            try:
                item = vec_queue.get(timeout=5)
            except Empty:
                continue

            if item is None:
                break

            vectors, metas = item

            filtered_vectors = []
            filtered_metas = []
            for vec, meta in zip(vectors, metas):
                chunk_id = meta.get("chunk_id") or meta.get("text")  # unique identifier
                if chunk_id not in processed_ids:
                    filtered_vectors.append(vec)
                    filtered_metas.append(meta)
                    processed_ids.add(chunk_id)

            if not filtered_vectors:
                continue  

            vectors_to_add = np.vstack(filtered_vectors)  
            store = store or FaissStore(dim=vectors_to_add.shape[1])
            store.add(vectors_to_add, filtered_metas)
            vectors_written += len(filtered_vectors)

            if vectors_written % cfg.log_every < len(filtered_vectors):
                rate = vectors_written / max(time.time() - start_time, 1e-6)
                logger.info(
                    "Indexing progress | vectors=%d | rate=%.1f vec/s",
                    vectors_written, rate
                )

            if vectors_written % cfg.save_every < len(filtered_vectors):
                store.save(index_path, meta_path)
                logger.info(
                    "Checkpoint saved | vectors=%d | path=%s",
                    vectors_written, index_path
                )

        if store is not None:
            store.save(index_path, meta_path)
            logger.info(
                "Final FAISS index saved | vectors=%d | runtime=%.1fs",
                vectors_written, time.time() - start_time
            )

        logger.info("FAISS writer thread finished")

