from pathlib import Path
from pprint import pprint

from Reasona.pipeline.retrieval_pipeline import RetrievalPipeline
from Reasona.entities.config_entity import RetrievalConfig


def main():
    cfg = RetrievalConfig(
        vector_store_dir=Path("artifacts/vectors"),  
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k=3,
    )

    pipeline = RetrievalPipeline(cfg)

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
            print(r.get("text", "")[:500])
            print("metadata:", r.get("metadata"))
            print("-" * 60)


if __name__ == "__main__":
    main()
