from pathlib import Path
import faiss
import pickle

class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add(self, vectors, metadata):
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def save(self, path: Path):
        
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, path: Path) -> "FaissStore":
        index_file = path / "index.faiss"
        meta_file = path / "meta.pkl"

        if not index_file.exists():
            raise FileNotFoundError("index.faiss not found")

        index = faiss.read_index(str(index_file))
        store = cls(index.d)
        store.index = index

        if meta_file.exists():
            with open(meta_file, "rb") as f:
                store.metadata = pickle.load(f)
        else:
            store.metadata = []

        return store
