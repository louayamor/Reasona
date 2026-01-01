from typing import Iterable
from Reasona.utils.logger import setup_logger
from Reasona.config.config_manager import ConfigurationManager
from Reasona.data.chunker import TextChunker
from Reasona.data.embedder import Embedder
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.pipeline.preprocess_pipeline import PreprocessPipeline

logger = setup_logger("indexing_pipeline", "logs/pipeline/indexing_pipeline.json")


class IndexingPipeline:
    def __init__(self):
        logger.info("Initializing IndexingPipeline()")

        cfg = ConfigurationManager()
        self.embed_cfg = cfg.get_embedding_config()

        self.vector_db_dir = self.embed_cfg.vector_store_dir
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        logger.info("=== INDEXING PIPELINE STARTED ===")

        preprocess_pipeline = PreprocessPipeline()
        dataset_stream = preprocess_pipeline.run()

        chunker = TextChunker(
            chunk_size=self.embed_cfg.chunk_size,
            chunk_overlap=self.embed_cfg.chunk_overlap,
        )
        embedder = Embedder(model_name=self.embed_cfg.embedding_model)

        store = None
        first_vector_dim = None

        for i, sample in enumerate(dataset_stream):
            chunks = chunker.chunk_text(sample["text"], metadata=sample.get("metadata"))
            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]

            vectors = embedder.embed(texts)

            if store is None:
                first_vector_dim = vectors.shape[1]
                store = FaissStore(dim=first_vector_dim)

            store.add(vectors, metadatas)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1} samples")

        if store:
            store.save(self.vector_db_dir)
            logger.info(f"Vector store saved to {self.vector_db_dir}")

        logger.info("=== INDEXING PIPELINE FINISHED ===")


if __name__ == "__main__":
    IndexingPipeline().run()
