import numpy as np
from typing import List, Dict, Optional, Callable, Iterable
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.data.embedder import Embedder


class Retriever:
    def __init__(self, store: FaissStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> List[Dict]:
        
        q_vec = self.embedder.embed([query]).astype("float32")
        distances, indices = self.store.index.search(q_vec, k)

        results = []
        for rank, idx in enumerate(indices[0]):
            item = {
                "text": self.store.texts[idx],
                "metadata": self.store.metadata[idx],
            }
            if filter_fn and not filter_fn(item):
                continue
            if return_scores:
                item["score"] = float(distances[0][rank])
            results.append(item)
        return results

    def retrieve_batch(
        self,
        queries: Iterable[str],
        k: int = 5,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> List[List[Dict]]:
        batch_results = []
        for q in queries:
            batch_results.append(self.retrieve(q, k=k, return_scores=return_scores, filter_fn=filter_fn))
        return batch_results
