from pathlib import Path
import pickle
import faiss
from Reasona.vectorstore.faiss_store import FaissStore

def main():
    store_dir = Path("artifacts/vectors")
    index_file = store_dir / "index.faiss"

    if not index_file.exists():
        print("FAISS index file not found!")
        return

    index = faiss.read_index(str(index_file))
    store = FaissStore(dim=index.d)
    store.index = index

    meta_path = store_dir / "meta.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            store.metadata = pickle.load(f)
    else:
        print("Metadata file not found, only vectors loaded.")
        store.metadata = []

    print(f"Loaded FAISS index with {store.ntotal} vectors")
    print("-" * 50)

    n_preview = min(10, store.ntotal)
    for i in range(n_preview):
        vector = store.index.reconstruct(i)
        meta = store.metadata[i] if i < len(store.metadata) else {}
        text = meta.get("text") or "<no text>"
        print(f"[{i}] text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"    vector snippet: {vector[:10]} ...")
        print("-" * 50)


if __name__ == "__main__":
    main()
