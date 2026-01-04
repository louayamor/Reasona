# Reasona/data/chunker.py
from typing import Dict, Iterable, Iterator
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/chunker.json")


class TextChunker:
    def __init__(self, chunk_size: int, overlap: int, log_every: int = 50_000):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.log_every = log_every
        self._chunks_processed = 0

    def chunk_item(self, item: dict):
        text = item.get("text", "")
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append({"text": " ".join(chunk_words), "_meta": item})
            start += self.chunk_size - self.overlap
        
        self._chunks_processed += len(chunks)
        # only log first chunk or every log_every chunks
        if self._chunks_processed <= len(chunks) or self._chunks_processed % self.log_every < len(chunks):
            logger.info(
                f"Chunked item | original_text_length={len(words)} words, total_chunks={len(chunks)}"
            )
        return chunks
