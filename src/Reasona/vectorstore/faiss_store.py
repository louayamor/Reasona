from pathlib import Path
import faiss
import pickle

class FaissStore:

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: list = []

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add(self, vectors, metadata: list):
        
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def save(self, index_path: Path, meta_path: Path):
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "FaissStore":
        
        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} not found")

        index = faiss.read_index(str(index_path))
        store = cls(index.d)
        store.index = index

        if meta_path.exists():
            with open(meta_path, "rb") as f:
                store.metadata = pickle.load(f)
        else:
            store.metadata = []

        return store

    def clear_metadata(self):
        
        self.metadata = []
