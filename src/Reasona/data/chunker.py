# Reasona/data/chunker.py
from typing import List, Dict
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/chunker.json")


class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(
            f"Initialized TextChunker(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = words[start:end]
            chunks.append(" ".join(chunk))
            start = end - self.chunk_overlap

        return chunks

    def chunk_dataset(self, dataset: List[Dict]) -> List[Dict]:
        all_chunks = []

        for item in dataset:
            base_text = f"{item['instruction']}\n{item['output']}"
            chunks = self.chunk_text(base_text)

            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "text": chunk,
                        "metadata": {
                            **item.get("metadata", {}),
                            "chunk_id": i,
                        },
                    }
                )

        logger.info(f"Generated {len(all_chunks)} text chunks")
        return all_chunks
