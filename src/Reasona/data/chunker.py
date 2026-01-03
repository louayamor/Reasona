# Reasona/data/chunker.py
from typing import Dict, Iterable, Iterator
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/chunker.json")


class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(
            "Initialized TextChunker(chunk_size=%d, overlap=%d)",
            chunk_size,
            chunk_overlap,
        )

    def chunk_text(self, text: str) -> Iterator[str]:
        if not text.strip():
            logger.warning("Received empty text for chunking")
            return

        words = text.split()
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            yield " ".join(chunk_words)
            chunk_idx += 1
            start = end - self.chunk_overlap

    def chunk_item(self, item: Dict) -> Iterator[Dict]:
       
        text = item.get("text", "").strip()
        if not text:
            logger.warning("Skipping item with empty text | metadata=%s", item.get("metadata"))
            return

        base_meta = item.get("metadata", {})
        total_chunks = 0

        for i, chunk in enumerate(self.chunk_text(text)):
            total_chunks += 1
            yield {
                "text": chunk,
                "metadata": {
                    **base_meta,
                    "chunk_id": i,
                },
            }

        logger.info(
            "Chunked item | original_text_length=%d words, total_chunks=%d, metadata=%s",
            len(text.split()), total_chunks, base_meta
        )

    def chunk_stream(self, items: Iterable[Dict]) -> Iterator[Dict]:
      
        for item in items:
            yield from self.chunk_item(item)
