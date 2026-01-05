import faiss
import pickle
from pathlib import Path


class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    @property
    def ntotal(self):
        return self.index.ntotal

    @property
    def dim(self):
        return self.index.d

    def add(self, vectors, metadata):
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: Path):
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "meta.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    @classmethod
    def load_from_dir(cls, path: Path) -> "FaissStore":
        index = faiss.read_index(str(path / "index.faiss"))
        store = cls(index.d)          
        store.index = index

        with open(path / "meta.pkl", "rb") as f:
            store.metadata = pickle.load(f)

        return store
