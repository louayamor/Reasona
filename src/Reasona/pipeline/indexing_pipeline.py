from Reasona.utils.logger import setup_logger
from Reasona.data.chunker import TextChunker
from Reasona.data.embedder import Embedder
from Reasona.vectorstore.faiss_store import FaissStore
from Reasona.pipeline.preprocess_pipeline import PreprocessPipeline
from Reasona.entities.config_entity import PreprocessConfig, IndexingConfig

logger = setup_logger("indexing_pipeline", "logs/pipeline/indexing_pipeline.json")


class IndexingPipeline:
    """
    Consumer pipeline.
    Consumes a preprocessing stream and builds a vector index.
    """

    def __init__(
        self,
        preprocess_cfg: PreprocessConfig,
        indexing_cfg: IndexingConfig,
    ):
        logger.info("Initializing IndexingPipeline (consumer)")

        self.preprocess_cfg = preprocess_cfg
        self.indexing_cfg = indexing_cfg

        self.vector_db_dir = indexing_cfg.vector_store_dir
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = TextChunker(
            chunk_size=indexing_cfg.chunk_size,
            chunk_overlap=indexing_cfg.chunk_overlap,
        )

        self.embedder = Embedder(model_name=indexing_cfg.embedding_model)

    def run(self) -> None:
        logger.info("=== INDEXING PIPELINE STARTED ===")

        # ---- producer ----
        preprocess = PreprocessPipeline(self.preprocess_cfg)
        stream = preprocess.stream()

        store: FaissStore | None = None
        processed = 0

        for sample in stream:
            chunks = self.chunker.chunk_text(
                sample["text"],
                metadata=sample.get("metadata"),
            )

            if not chunks:
                continue

            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]

            vectors = self.embedder.embed(texts)

            if store is None:
                store = FaissStore(dim=vectors.shape[1])

            store.add(vectors, metadatas)

            processed += 1
            if processed % 100 == 0:
                logger.info(f"Indexed {processed} streamed samples")

        if store is not None:
            store.save(self.vector_db_dir)
            logger.info(f"Vector store saved to {self.vector_db_dir}")

        logger.info("=== INDEXING PIPELINE FINISHED ===")
