# Reasona/inference/reranker.py

from typing import List, Dict
import torch
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder reranker.
    Pure scoring logic, no retrieval.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        if not chunks:
            return []

        pairs = [(query, c["text"]) for c in chunks]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
        )

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks[:top_k]
