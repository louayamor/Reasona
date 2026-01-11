from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple
import numpy as np
import faiss

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
    Retrieval pipeline using:
    - GPU embeddings
    - FAISS IVF mmap index
    - SQLite-backed metadata
    """

    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.debug = cfg.debug
        self.vector_store_dir: Path = cfg.vector_store_dir

        index_path = self.vector_store_dir / "index.faiss"
        db_path = self.vector_store_dir / "metadata.db"

        self.embedder = Embedder(
            model_name=cfg.embedding_model,
            batch_size=1,              
            device="cuda",
            log_every=cfg.log_every,
        )

        dummy = self.embedder.embed(["_dim_check_"])
        embedding_dim = dummy.shape[1]

        self.store = FaissStore(
            dim=embedding_dim,
            index_path=index_path,
            db_path=db_path,
            nprobe=cfg.nprobe,
        )
        self.store.load(mmap=True)

        logger.info(
            "FAISS state | trained=%s | ntotal=%d | nprobe=%d",
            self.store.index.is_trained,
            self.store.index.ntotal,
            self.store.index.nprobe,
        )

        logger.info(
            "FAISS store loaded | vectors=%d",
            self.store.ntotal,
        )

        self.retriever = Retriever()

        self._cache: Dict[
            Tuple[str, int, bool, int],
            List[Dict],
        ] = {}

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> List[Dict]:

        top_k = top_k or self.cfg.top_k

        cache_key = (
            query_text,
            top_k,
            return_scores,
            id(filter_fn),
        )

        if cache_key in self._cache:
            return self._cache[cache_key]

        query_vector = self.embedder.embed([query_text]).astype("float32")

        results = self.retriever.retrieve(
            query_vector=query_vector,
            k=top_k,
            return_scores=return_scores,
            filter_fn=filter_fn,
            index=self.store
        )

        self._cache[cache_key] = results
        return results

    def run_query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> Dict:

        chunks = self.query(query_text, top_k=top_k)
        prompt_input = "\n\n".join(c["text"] for c in chunks)

        return {
            "query": query_text,
            "chunks": chunks,
            "prompt_input": prompt_input,
        }

    def run(self):
        print("Retrieval pipeline ready. Type a query (or 'exit').")

        while True:
            try:
                query_text = input(">> ").strip()
                if query_text.lower() in {"exit", "quit"}:
                    break
                if not query_text:
                    continue

                result = self.run_query(
                    query_text,
                    top_k=self.cfg.top_k,
                )

                chunks = result["chunks"]
                print(f"Retrieved {len(chunks)} chunks")

                if self.debug and chunks:
                    top = chunks[0]
                    print(f"Top score : {top.get('score')}")
                    print(f"Preview   : {top['text'][:200]}...")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.exception("Retrieval error")
                print(f"Error: {e}")

    @staticmethod
    def filter_by_source(source: str) -> Callable[[Dict], bool]:
        return lambda meta: meta.get("source") == source
