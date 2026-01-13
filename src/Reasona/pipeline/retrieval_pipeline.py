from pathlib import Path
from typing import List, Dict, Callable, Optional

import faiss
import numpy as np

from Reasona.data.embedder import Embedder
from Reasona.entities.config_entity import RetrievalConfig
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.inference.retriever import Retriever
from Reasona.utils.logger import setup_logger

faiss.omp_set_num_threads(1)

logger = setup_logger(
    "retrieval_pipeline",
    "logs/pipeline/retrieval_pipeline.json",
)


class RetrievalPipeline:
    """
    Stateless retrieval pipeline with detailed logging.
    API-first design (Flask / FastAPI ready).
    """

    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        logger.info("Initializing RetrievalPipeline...")

        vector_store_dir = Path(cfg.vector_store_dir)
        self.index_path = vector_store_dir / "index.faiss"
        self.db_path = vector_store_dir / "metadata.db"

        if not self.index_path.exists() or not self.db_path.exists():
            logger.error("FAISS index or metadata DB not found: %s, %s", self.index_path, self.db_path)
            raise FileNotFoundError(
                "FAISS index or metadata DB not found. Run indexing first."
            )

        logger.info("Loading embedder model: %s", cfg.embedding_model)
        self.embedder = Embedder(
            model_name=cfg.embedding_model,
            batch_size=1,
            device="cuda",
            log_every=cfg.log_every,
        )

        logger.info("Loading FAISS store from %s", self.index_path)
        self.store = FaissStore(
            dim=None,
            index_path=self.index_path,
            db_path=self.db_path,
            nprobe=cfg.nprobe,
            mmap=getattr(cfg, "mmap", True),
        )
        self.store.load()
        logger.info(
            "FAISS store loaded | ntotal=%d | nprobe=%d | mmap=%s",
            self.store.count_vectors(),
            self.store.index.nprobe,
            self.store.mmap,
        )

        self.retriever = Retriever()
        logger.info("Retriever initialized")

    def execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> Dict[str, object]:
        """
        Execute retrieval with logging.

        Returns:
        {
            query: str
            chunks: List[Dict]
            prompt_input: str
            stats: Dict
        }
        """
        logger.info("Executing retrieval for query: '%s'", query)

        if not query or not query.strip():
            logger.warning("Empty query received")
            raise ValueError("query must be non-empty")

        top_k = top_k or self.cfg.top_k
        logger.info("Top-k set to %d", top_k)

        logger.info("Embedding query text")
        query_vector = self.embedder.embed([query]).astype("float32")

        logger.info("Retrieving top %d chunks from FAISS", top_k)
        chunks = self.retriever.retrieve(
            query_vector=query_vector,
            k=top_k,
            return_scores=return_scores,
            filter_fn=filter_fn,
            index=self.store,
        )
        logger.info("Retrieved %d chunks", len(chunks))

        prompt_input = self._build_prompt(chunks)
        logger.info("Built prompt of length %d characters", len(prompt_input))

        return {
            "query": query,
            "chunks": chunks,
            "prompt_input": prompt_input,
            "stats": {
                "top_k": top_k,
                "num_chunks": len(chunks),
            },
        }

    @staticmethod
    def _build_prompt(chunks: List[Dict]) -> str:
        return "\n\n".join(c["text"] for c in chunks)

    @staticmethod
    def filter_by_source(source: str) -> Callable[[Dict], bool]:
        return lambda meta: meta.get("source") == source
