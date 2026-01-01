from pathlib import Path
from typing import List, Dict

from Reasona.utils.logger import setup_logger
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.vectorstore.retriever import Retriever
from Reasona.data.embedder import Embedder
from Reasona.entities.config_entity import IndexingConfig

logger = setup_logger("retrieval_pipeline", "logs/pipeline/retrieval_pipeline.json")


class RetrievalPipeline:
    """
    Retrieval pipeline for querying indexed vectors.
    Depends on the FAISS vector store built by IndexingPipeline.
    """

    def __init__(self, cfg: IndexingConfig):
        self.cfg = cfg
        self.vector_store_dir: Path = cfg.vector_store_dir

        self.store = FaissStore(dim=0)  
        self.store.load(self.vector_store_dir)
        logger.info(f"Loaded FAISS vector store from {self.vector_store_dir}")

        self.embedder = Embedder(cfg.embedding_model)

        self.retriever = Retriever(self.store, self.embedder)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:

        logger.info(f"Running retrieval for query: {query_text[:50]}...")
        results = self.retriever.retrieve(query_text, k=top_k)
        logger.info(f"Retrieved {len(results)} results")
        return results
