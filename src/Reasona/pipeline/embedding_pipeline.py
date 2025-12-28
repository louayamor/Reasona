import json
from pathlib import Path
from typing import Iterable

from Reasona.config.config_manager import ConfigurationManager
from Reasona.data.chunker import TextChunker
from Reasona.data.embedder import Embedder
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.utils.logger import setup_logger

logger = setup_logger("embedding_pipeline", "logs/pipeline/embedding_pipeline.json")

class EmbeddingPipeline:
    def __init__(self):
        logger.info("Initializing EmbeddingPipeline()")

        cfg = ConfigurationManager()
        self.embed_cfg = cfg.get_embedding_config()

        self.dataset_path = self.embed_cfg.dataset_path
        self.vector_db_dir = self.embed_cfg.vector_db_dir

        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self) -> Iterable[dict]:
        logger.info(f"Loading dataset from {self.dataset_path}")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Skipping invalid JSON at line {line_no}: {e}"
                    )

    def run(self):
        logger.info("=== EMBEDDING PIPELINE STARTED ===")

        # 1. Load dataset
        dataset = list(self.load_dataset())
        logger.info(f"Loaded {len(dataset)} samples")

        # 2. Chunk
        chunker = TextChunker(
            chunk_size=self.embed_cfg.chunk_size,
            chunk_overlap=self.embed_cfg.chunk_overlap,
        )
        chunks = chunker.chunk_dataset(dataset)
        logger.info(f"Generated {len(chunks)} chunks")

        texts = [c["text"] for c in chunks]
        metadata = [c["metadata"] for c in chunks]

        # 3. Embed
        embedder = Embedder(model_name=self.embed_cfg.embedding_model)
        vectors = embedder.embed(texts)
        logger.info(f"Generated embeddings: {vectors.shape}")

        # Store in FAISS
        store = FaissStore(dim=vectors.shape[1])
        store.add(vectors, metadata)
        store.save(self.vector_db_dir)

        logger.info(f"Vector store saved to {self.vector_db_dir}")
        logger.info("=== EMBEDDING PIPELINE FINISHED ===")


if __name__ == "__main__":
    EmbeddingPipeline().run()
