from pathlib import Path
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from Reasona.utils.logger import setup_logger
from Reasona.pipeline.preprocess_pipeline import PreprocessPipeline
from Reasona.data.chunker import TextChunker
from Reasona.data.embedder import Embedder
from Reasona.vectorstore.faiss_store import FaissStore

logger = setup_logger("indexing_pipeline", "logs/pipeline/indexing_pipeline.json")


class IndexingPipeline:
    """
    Streaming consumer pipeline.
    Consumes preprocessed samples and builds a FAISS vector store.
    """

    def __init__(
        self,
        preprocess_pipeline: PreprocessPipeline,
        chunker: TextChunker,
        embedder: Embedder,
        vector_db_dir: Path,
        workers: int = 2,
        queue_size: int = 100,
    ):
        self.preprocess_pipeline = preprocess_pipeline
        self.chunker = chunker
        self.embedder = embedder
        self.vector_db_dir = Path(vector_db_dir)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

        self.workers = workers
        self.queue = Queue(maxsize=queue_size)

    def run(self) -> None:
        logger.info("=== INDEXING PIPELINE STARTED ===")

        store: Optional[FaissStore] = None

        # ---------------- PRODUCER ----------------
        def producer():
            for sample in self.preprocess_pipeline.stream():
                chunks = self.chunker.chunk_text(
                    sample["text"],
                    metadata=sample.get("metadata"),
                )
                if chunks:
                    self.queue.put(chunks)

            # signal termination
            for _ in range(self.workers):
                self.queue.put(None)

        # ---------------- CONSUMER ----------------
        def consumer():
            nonlocal store

            while True:
                chunks = self.queue.get()
                if chunks is None:
                    break

                texts = [c["text"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]

                vectors = self.embedder.embed(texts)

                if store is None:
                    store = FaissStore(dim=vectors.shape[1])

                store.add(vectors, metadatas)

        # start producer
        Thread(target=producer, daemon=True).start()

        # start consumers
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for _ in range(self.workers):
                executor.submit(consumer)

        if store is not None:
            store.save(self.vector_db_dir)
            logger.info(f"Vector store saved to {self.vector_db_dir}")

        logger.info("=== INDEXING PIPELINE FINISHED ===")
