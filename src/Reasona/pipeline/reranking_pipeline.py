# Reasona/pipeline/reranking_pipeline.py

from typing import Dict, Optional
from Reasona.pipeline.retrieval_pipeline import RetrievalPipeline
from Reasona.config.config_manager import ConfigurationManager
from Reasona.inference.reranker import Reranker
from Reasona.utils.logger import setup_logger


logger = setup_logger(
    "reranking_pipeline",
    "logs/pipeline/reranking_pipeline.json",
)


class RerankingPipeline:
    """
    Middle-ground pipeline:
    Retrieval → (optional) Reranking → Inference-ready output
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        cfg_manager: ConfigurationManager,
    ):
        self.retrieval_pipeline = retrieval_pipeline

        cfg = cfg_manager.get_reranking_config()
        self.enabled = cfg.enabled
        self.top_k = cfg.top_k

        self.reranker = None
        if self.enabled:
            self.reranker = Reranker(
                model_name=cfg.model,
                batch_size=cfg.batch_size,
            )

        logger.info(
            "Reranker enabled=%s model=%s",
            self.enabled,
            cfg.model if self.enabled else "none",
        )

    def run_query(
        self,
        query_text: str,
        retrieval_top_k: Optional[int] = None,
    ) -> Dict:
        """
        Final retrieval + reranking entrypoint
        """

        retrieval_result = self.retrieval_pipeline.run_query(
            query_text,
            top_k=retrieval_top_k,
        )

        chunks = retrieval_result["chunks"]

        if self.enabled and self.reranker:
            chunks = self.reranker.rerank(
                query=query_text,
                chunks=chunks,
                top_k=self.top_k,
            )

        prompt_input = "\n\n".join(c["text"] for c in chunks)

        return {
            "query": query_text,
            "chunks": chunks,
            "prompt_input": prompt_input,
        }
