from pathlib import Path
from Reasona.pipeline.retrieval_pipeline import RetrievalPipeline
from Reasona.entities.config_entity import RetrievalConfig
from Reasona.vectorstore.faiss_store import FaissStore

def main():
    # Resolve the vector directory relative to the project root
    project_root = Path(__file__).parent.parent
    vector_dir = project_root / "artifacts" / "vectors"

    index_path = vector_dir / "index.faiss"
    meta_path = vector_dir / "meta.pkl"

    cfg = RetrievalConfig(
        vector_store_dir=vector_dir,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k=3,
        engine="vector_search",
    )

    pipeline = RetrievalPipeline(cfg)

    # Directly load the FAISS store from absolute paths
    pipeline.store = FaissStore.load(index_path, meta_path)

    print("\nRetrieval pipeline ready.")
    print("Type a query (or 'exit'):\n")

    while True:
        query = input(">> ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        results = pipeline.query(query)

        print("\nTop results:\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r.get('score')}")
            print(r.get("text", "<no text>")[:500])
            print("metadata:", r.get("metadata", {}))
            print("-" * 60)


if __name__ == "__main__":
    main()
