from typing import List, Dict, Callable, Optional, Iterable
import numpy as np
from Reasona.vectorstore.faiss_store import FaissStore


class Retriever:
    """
    Handles retrieval logic for the disk-backed FaissStore.
    """

    def retrieve(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        index: FaissStore = None,
    ) -> List[Dict]:

        if index is None:
            raise ValueError("FaissStore instance must be provided as 'index'.")

        query_vector = np.atleast_2d(query_vector).astype("float32")

        distances, metas = index.search(query_vector, k)

        if not metas:
            return []

        distances = np.asarray(distances).flatten()

        output: List[Dict] = []

        for rank, meta in enumerate(metas):
            if filter_fn and not filter_fn(meta):
                continue

            item = {
                "text": meta.get("text", ""),
                "metadata": meta,
            }

            if return_scores:
                # L2 distance → similarity score
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

        return [
            self.retrieve(
                vec,
                k=k,
                return_scores=return_scores,
                filter_fn=filter_fn,
                index=index,
            )
            for vec in query_vectors
        ]
