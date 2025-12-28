from sentence_transformers import SentenceTransformer
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/embedder.json")


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]):
        logger.info(f"Embedding {len(texts)} chunks")
        return self.model.encode(texts, show_progress_bar=True)
