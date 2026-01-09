# retrieval_pipeline.py
from pathlib import Path
from typing import List, Dict, Callable, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from Reasona.data.embedder import Embedder
from Reasona.entities.config_entity import RetrievalConfig
from Reasona.vectorstore.retriever import Retriever
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.utils.logger import setup_logger

logger = setup_logger("retrieval_pipeline", "logs/pipeline/retrieval_pipeline.json")


class RetrievalPipeline:
    """
    Orchestrates:
    - Loading FAISS store
    - Embedding queries
    - Using Retriever for scoring/filtering
    - Batch and parallel batch queries
    - RAG-ready output
    """

    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.vector_store_dir: Path = cfg.vector_store_dir
        self.max_workers: int = getattr(cfg, "max_workers", 4)
        self.use_cache: bool = getattr(cfg, "use_cache", True)

        index_path = self.vector_store_dir / "index.faiss"
        db_path = self.vector_store_dir / "metadata.db"

        self.embedder = Embedder(cfg.embedding_model)
        embedding_dim = getattr(cfg, "embedding_dim", None)
        if embedding_dim is None:
            dummy_vec = self.embedder.embed(["test"])
            embedding_dim = dummy_vec.shape[1]
            logger.info(f"Inferred embedding dimension: {embedding_dim}")

        self.store = FaissStore(dim=embedding_dim, index_path=index_path, db_path=db_path)
        self.store.load()
        logger.info(f"Loaded FAISS store with {self.store.ntotal} vectors")

        self.retriever = Retriever()

        self._cache: Dict[str, List[Dict]] = {}

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        use_cache: Optional[bool] = None,
    ) -> List[Dict]:
        top_k = top_k or self.cfg.top_k
        use_cache = self.use_cache if use_cache is None else use_cache

        if use_cache and query_text in self._cache:
            logger.info("Cache hit for query")
            return self._cache[query_text]

        query_vector = self.embedder.embed([query_text])[0]

        results = self.retriever.retrieve(
            query_vector=query_vector,
            k=top_k,
            return_scores=return_scores,
            filter_fn=filter_fn,
            index=self.store
        )

        if use_cache:
            self._cache[query_text] = results

        return results

    def query_batch(
        self,
        queries: Iterable[str],
        top_k: Optional[int] = None,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        use_cache: Optional[bool] = None,
    ) -> List[List[Dict]]:
        use_cache = self.use_cache if use_cache is None else use_cache
        query_list = list(queries)
        query_vectors = [self.embedder.embed([q])[0] for q in query_list]

        results = self.retriever.retrieve_batch(
            query_vectors=query_vectors,
            k=top_k or self.cfg.top_k,
            return_scores=return_scores,
            filter_fn=filter_fn,
            index=self.store
        )
        return results

    def query_batch_parallel(
        self,
        queries: Iterable[str],
        top_k: Optional[int] = None,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        use_cache: Optional[bool] = None,
    ) -> List[List[Dict]]:
        queries = list(queries)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                executor.map(
                    lambda q: self.query(q, top_k, return_scores, filter_fn, use_cache),
                    queries
                )
            )
        return results
    
    def run(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> Dict:
        chunks = self.query(query_text, top_k=top_k, filter_fn=filter_fn)
        prompt_input = "\n\n".join([c["text"] for c in chunks])
        return {"query": query_text, "chunks": chunks, "prompt_input": prompt_input}

    @staticmethod
    def filter_by_source(source: str) -> Callable[[Dict], bool]:
        return lambda chunk: chunk.get("source") == source
