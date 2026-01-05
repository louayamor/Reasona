from pathlib import Path
import numpy as np
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.vectorstore.faiss_store_manager import FaissStoreManager

def main():
    store_dir = Path("artifacts/vectors")
    manager = FaissStoreManager(store_dir)

    index = manager.load_latest_index()
    if index is None:
        print("No FAISS index found yet!")
        return

    store = FaissStore(dim=index.d)
    store.index = index

    meta_path = store_dir / "meta.pkl"
    if meta_path.exists():
        import pickle
        with open(meta_path, "rb") as f:
            store.metadata = pickle.load(f)
    else:
        print("Metadata file not found, only vectors loaded.")

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
