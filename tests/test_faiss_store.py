from pathlib import Path
import sqlite3
import json
import numpy as np

from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.data.embedder import Embedder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = PROJECT_ROOT / "artifacts/vectors/index.faiss"
DB_PATH = PROJECT_ROOT / "artifacts/vectors/metadata.db"

DIM = 384


def show_first_10_samples(db_path: Path):
    print("\n=== FIRST 10 STORED SAMPLES ===")
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, data FROM metadata ORDER BY id ASC LIMIT 10")

        for row_id, data in cur.fetchall():
            meta = json.loads(data)
            text = meta.get("text", "<no text>")[:300]
            print(f"[{row_id}] {text}")
            print("-" * 60)


def run_search_tests(store: FaissStore):
    embedder = Embedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda",
        batch_size=16,
        log_every=0,
    )

    queries = [
        "American Civil War soldier",
        "Japanese anime soundtrack composer",
        "small village in Iran",
    ]

    for i, query in enumerate(queries):
        q_vec = embedder.embed([query]).astype("float32")

        distances, results = store.search(q_vec, k=5)

        print(f"\n[QUERY {i}] {query}")
        if not results:
            print("  No results found.")
            continue

        for rank, (dist, meta) in enumerate(zip(distances, results)):
            snippet = meta.get("text", "<no text>")[:120]
            print(f"  [{rank}] dist={dist:.4f} | {snippet}")


def main():
    if not INDEX_PATH.exists() or not DB_PATH.exists():
        print("Index or metadata DB not found.")
        return

    store = FaissStore(
        dim=DIM,
        index_path=INDEX_PATH,
        db_path=DB_PATH,
        max_vectors=3_100_000,
    )

    # 🔹 mmap-enabled load (RETRIEVAL ONLY)
    store.load(mmap=True)

    print(f"Loaded FAISS index with {store.ntotal} vectors (mmap enabled)")

    show_first_10_samples(DB_PATH)
    run_search_tests(store)

    store.close()


if __name__ == "__main__":
    main()
