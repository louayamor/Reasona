# retriever.py
from typing import List, Dict, Callable, Optional, Iterable
import numpy as np
from Reasona.vectorstore.faiss_store import FaissStore


class Retriever:
    """
    Handles retrieval logic given a FaissStore and query vectors.
    - Does not manage embedding or FAISS initialization.
    - Supports single query, batch queries, and filtering.
    """

    def __init__(self):
        pass  

    def retrieve(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        index: FaissStore = None,
    ) -> List[Dict]:
        """
        Retrieve top-k items for a single query vector.
        `index` must be a FaissStore instance.
        """
        if index is None:
            raise ValueError("FaissStore instance must be provided as 'index'.")

        distances, results_meta = index.search(query_vector, k)
        output: List[Dict] = []

        for rank, meta in enumerate(results_meta):
            if not meta:
                continue
            if filter_fn and not filter_fn(meta):
                continue

            item = {
                "text": str(meta.get("text") or meta.get("content") or ""),
                "metadata": meta,
            }
            if return_scores:
                item["score"] = float(1.0 / (1.0 + distances[rank]))
            output.append(item)

        return output

    def retrieve_batch(
        self,
        query_vectors: Iterable[np.ndarray],
        k: int = 5,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        index: FaissStore = None,
    ) -> List[List[Dict]]:
        """
        Retrieve top-k items for multiple query vectors.
        """
        if index is None:
            raise ValueError("FaissStore instance must be provided as 'index'.")

        results = [
            self.retrieve(
                vec,
                k=k,
                return_scores=return_scores,
                filter_fn=filter_fn,
                index=index
            )
            for vec in query_vectors
        ]
        return results
