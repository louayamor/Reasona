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

        results: List[Dict] = []

        for rank, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            meta = self.store.metadata[idx]

            text = str(meta.get("text") or meta.get("content") or "")

            item = {
                "text": text,
                "metadata": meta,
            }

            if filter_fn and not filter_fn(meta):
                continue

            if return_scores:
                item["score"] = float(1.0 / (1.0 + distances[0][rank]))

            results.append(item)

        return results

    def retrieve_batch(
        self,
        queries: Iterable[str],
        k: int = 5,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> List[List[Dict]]:

        return [
            self.retrieve(q, k=k, return_scores=return_scores, filter_fn=filter_fn)
            for q in queries
        ]
