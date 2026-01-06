import time
import pickle
from pathlib import Path
from queue import Queue, Full, Empty
from threading import Thread
from typing import Optional
from Reasona.data.embedder import Embedder
from Reasona.data.chunker import TextChunker
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.config.config_manager import IndexingConfig
from Reasona.utils.logger import setup_logger
import glob

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

        self.log_every = cfg.log_every or 50_000
        self.save_every = cfg.save_every or 100_000  

        self.store: Optional[FaissStore] = None

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

    def _writer_worker(self):
        logger.info("FAISS writer started")

        vectors_written = 0
        all_metadata = []  
        start_time = time.time()

        index_files = sorted(glob.glob(str(self.vector_db_dir / "*.index")))
        if index_files:
            latest_index_path = index_files[-1]
            self.store = FaissStore(dim=384)  
            self.store.load(latest_index_path)
            meta_path = Path(latest_index_path).with_suffix(".pkl")
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    all_metadata = pickle.load(f)
            logger.info("Loaded FAISS index from %s with %d vectors", latest_index_path, len(all_metadata))
        else:
            self.store = None

        while True:
            try:
                item = self.vec_queue.get(timeout=5)
            except Empty:
                continue
            if item is None:
                break

            vectors, metas = item

            if self.store is None:
                dim = vectors.shape[1]
                self.store = FaissStore(dim=dim)
                logger.info("FAISS initialized | dim=%d", dim)

            self.store.add(vectors, metas)
            all_metadata.extend(metas)
            vectors_written += vectors.shape[0]

            if vectors_written % self.log_every < vectors.shape[0]:
                rate = vectors_written / max(time.time() - start_time, 1e-6)
                logger.info(
                    "Indexing progress | vectors=%d total=%d | rate=%.1f vec/s",
                    vectors_written, len(self.store.metadata), rate
                )

            if vectors_written % self.save_every < vectors.shape[0]:
                index_path = self.vector_db_dir / f"index_{int(time.time())}.index"
                self.store.save(index_path)
                meta_path = index_path.with_suffix(".pkl")
                with open(meta_path, "wb") as f:
                    pickle.dump(all_metadata, f)
                logger.info("Incremental save | path=%s | total_vectors=%d", index_path, len(all_metadata))

        if self.store is not None:
            index_path = self.vector_db_dir / f"index_{int(time.time())}.index"
            self.store.save(index_path)
            meta_path = index_path.with_suffix(".pkl")
            with open(meta_path, "wb") as f:
                pickle.dump(all_metadata, f)
            logger.info("Vector store saved | path=%s total_vectors=%d | runtime=%.1fs",
                        index_path, len(all_metadata), time.time() - start_time)

        logger.info("FAISS writer finished")
