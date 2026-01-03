from pathlib import Path
from typing import List, Dict, Callable, Optional, Iterable

from Reasona.utils.logger import setup_logger
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.data.embedder import Embedder
from Reasona.vectorstore.retriever import Retriever
from Reasona.entities.config_entity import RetrievalConfig

logger = setup_logger("retrieval_pipeline", "logs/pipeline/retrieval_pipeline.json")


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline for querying indexed vectors.
    """

    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.vector_store_dir: Path = cfg.vector_store_dir

        # Load FAISS vector store
        self.store = FaissStore.load_from_dir(self.vector_store_dir)
        logger.info(f"Loaded FAISS vector store from {self.vector_store_dir} (dim={self.store.dim})")

        # Embedder
        self.embedder = Embedder(cfg.embedding_model)

        # Retriever
        self.retriever = Retriever(self.store, self.embedder)

    # ---------------- Public API ----------------

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None
    ) -> List[Dict]:
        """Single-query retrieval."""
        top_k = top_k or self.cfg.top_k
        logger.info(f"Running retrieval for query: {query_text[:50]}...")
        results = self.retriever.retrieve(
            query_text, k=top_k, return_scores=return_scores, filter_fn=filter_fn
        )
        logger.info(f"Retrieved {len(results)} results")
        return results

    def query_batch(
        self,
        queries: Iterable[str],
        top_k: Optional[int] = None,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None
    ) -> List[List[Dict]]:
        queries = list(queries)
        top_k = top_k or self.cfg.top_k
        logger.info(f"Running batch retrieval for {len(queries)} queries...")
        results = self.retriever.retrieve_batch(
            queries, k=top_k, return_scores=return_scores, filter_fn=filter_fn
        )
        logger.info("Batch retrieval finished")
        return results
