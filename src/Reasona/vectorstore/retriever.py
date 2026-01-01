import numpy as np

class Retriever:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5):
        # Embed query
        q_vec = self.embedder.embed([query])
        q_vec = np.array(q_vec).astype("float32")

        distances, indices = self.store.index.search(q_vec, k)

        results = []
        for idx in indices[0]:
            results.append({
                "text": self.store.texts[idx],       
                "metadata": self.store.metadata[idx],
            })
        return results
