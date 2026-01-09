from typing import Dict, Iterable, Iterator
from uuid import uuid4

from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/chunker.json")


class TextChunker:
    def __init__(self, chunk_size: int, overlap: int, log_every: int = 50_000):
        assert overlap < chunk_size, "overlap must be < chunk_size"

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.log_every = log_every
        self._chunks_processed = 0

    def chunk_item(self, item: Dict) -> Iterator[Dict]:
        text = item.get("text", "")
        if not text.strip():
            return

        words = text.split()
        start = 0

        while start < len(words):
            end = start + self.chunk_size

            yield {
                "id": str(uuid4()),               
                "text": " ".join(words[start:end]),
                "source": item.get("source"),
                "original_id": item.get("id"),
            }

            start += self.chunk_size - self.overlap
            self._chunks_processed += 1

            if self._chunks_processed % self.log_every == 0:
                logger.info(
                    "Chunking progress | total_chunks=%d",
                    self._chunks_processed,
                )

    def chunk_stream(self, items: Iterable[Dict]) -> Iterator[Dict]:
        for item in items:
            yield from self.chunk_item(item)
