from typing import List, Dict
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/chunker.json")


class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - self.overlap

        return chunks

    def chunk_dataset(self, dataset: List[Dict]) -> List[Dict]:
        logger.info("Starting chunking stage")
        chunked = []

        for item in dataset:
            text = f"{item['instruction']}\n{item['output']}"
            chunks = self.chunk_text(text)

            for idx, chunk in enumerate(chunks):
                chunked.append({
                    "text": chunk,
                    "metadata": {
                        **item.get("metadata", {}),
                        "chunk_id": idx,
                    }
                })

        logger.info(f"Chunking completed: {len(chunked)} chunks created")
        return chunked
