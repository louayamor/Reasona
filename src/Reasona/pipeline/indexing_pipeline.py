from pathlib import Path
from queue import Queue
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from Reasona.utils.logger import setup_logger
from Reasona.data.embedder import Embedder
from Reasona.vectorstore.faiss_store import FaissStore

logger = setup_logger("indexing_pipeline", "logs/pipeline/indexing_pipeline.json")


class IndexingPipeline:
    
    def __init__(
        self,
        embedder: Embedder,
        vector_db_dir: Path,
        workers: int = 2,
        queue_size: int = 100,
        batch_size: int = 32,
    ):
        self.embedder = embedder
        self.vector_db_dir = Path(vector_db_dir)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

        self.workers = workers
        self.batch_size = batch_size

        self.queue = Queue(maxsize=queue_size)
        self.store: Optional[FaissStore] = None
        self._faiss_lock = Lock()
        self._consumer_threads: List[Thread] = []

    # public methods
    def start(self):
        """Start consumer threads."""
        logger.info("=== INDEXING PIPELINE STARTED ===")
        for i in range(self.workers):
            t = Thread(target=self._consumer, name=f"_consumer-{i}", daemon=True)
            t.start()
            self._consumer_threads.append(t)

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Add preprocessed chunks to the queue."""
        self.queue.put(chunks)

    def stop(self):
        # signal termination
        for _ in range(self.workers):
            self.queue.put(None)

        for t in self._consumer_threads:
            t.join()

        if self.store is not None:
            self.store.save(self.vector_db_dir)
            logger.info(f"Vector store saved to {self.vector_db_dir}")

        logger.info("=== INDEXING PIPELINE FINISHED ===")

    # private methods
    def _consumer(self):
        """Worker that embeds chunks and adds them to FAISS safely."""
        batch_texts: List[str] = []
        batch_metadata: List[Dict[str, Any]] = []

        while True:
            chunks = self.queue.get()
            if chunks is None:
                if batch_texts:
                    self._add_to_faiss(batch_texts, batch_metadata)
                break

            if isinstance(chunks, dict):
                chunks = [chunks]

            for c in chunks:
                batch_texts.append(c["text"])
                batch_metadata.append(c.get("metadata", {}))

                if len(batch_texts) >= self.batch_size:
                    self._add_to_faiss(batch_texts, batch_metadata)
                    batch_texts, batch_metadata = [], []

        if batch_texts:
            self._add_to_faiss(batch_texts, batch_metadata)

    def _add_to_faiss(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        
        vectors = self.embedder.embed(texts)

        with self._faiss_lock:
            if self.store is None:
                self.store = FaissStore(dim=vectors.shape[1])
            self.store.add(vectors, metadatas)

        logger.info(f"Embedded & indexed batch of {len(texts)} chunks")
