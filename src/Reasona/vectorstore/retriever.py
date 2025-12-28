import numpy as np


class Retriever:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5):
        q_vec = self.embedder.embed([query])
        distances, indices = self.store.index.search(
            np.array(q_vec).astype("float32"), k
        )

        return [self.store.metadata[i] for i in indices[0]]
