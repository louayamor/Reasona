from typing import Dict, List, Iterable, Iterator
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/chunker.json")


class TextChunker:
    

    def __init__(self, chunk_size: int, overlap: int, log_every: int = 50_000):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.log_every = log_every
        self._chunks_processed = 0

    def chunk_item(self, item: Dict) -> List[Dict]:
        
        text = item.get("text", "")
        if not text.strip():
            logger.warning("Skipping empty text for sample | keys=%s", list(item.keys()))
            return []

        words = text.split()
        chunks: List[Dict] = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append({
                "text": " ".join(chunk_words),
                "_metadata": item  
            })
            start += self.chunk_size - self.overlap

        self._chunks_processed += len(chunks)
        if self._chunks_processed <= len(chunks) or self._chunks_processed % self.log_every < len(chunks):
            logger.info(
                "Chunked item | original_text_length=%d words, total_chunks=%d",
                len(words), len(chunks)
            )

        return chunks

    def chunk_stream(self, items: Iterable[Dict]) -> Iterator[Dict]:
       
        for item in items:
            yield from self.chunk_item(item)
