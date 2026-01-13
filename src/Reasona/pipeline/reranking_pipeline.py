from typing import Dict, List, Optional

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
    Stateless middle-layer pipeline:
    Retrieval → Optional reranking
    Designed for API / Flask usage
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        cfg_manager: ConfigurationManager,
    ):
        logger.info("Initializing RerankingPipeline...")
        self.retrieval_pipeline = retrieval_pipeline

        cfg = cfg_manager.get_reranking_config()

        self.enabled: bool = cfg.enabled
        self.top_k: int = cfg.top_k

        self.reranker: Optional[Reranker] = None
        if self.enabled:
            logger.info("Loading Reranker model: %s", cfg.model)
            self.reranker = Reranker(
                model_name=cfg.model,
                batch_size=cfg.batch_size,
            )
            logger.info("Reranker loaded successfully")

        logger.info(
            "RerankingPipeline initialized | enabled=%s | model=%s | top_k=%d",
            self.enabled,
            cfg.model if self.enabled else "none",
            self.top_k,
        )

    def execute(
        self,
        query: str,
        retrieval_top_k: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Execute retrieval + optional reranking with logging.

        Returns:
        {
            query: str
            chunks: List[Dict]
            prompt_input: str
        }
        """
        logger.info("Executing RerankingPipeline for query: '%s'", query)

        # Step 1: Retrieve chunks
        retrieval_result = self.retrieval_pipeline.execute(
            query=query,
            top_k=retrieval_top_k,
        )
        chunks: List[Dict] = retrieval_result["chunks"]
        logger.info(
            "Retrieved %d chunks from RetrievalPipeline | top_k=%s",
            len(chunks),
            retrieval_top_k or self.retrieval_pipeline.cfg.top_k,
        )

        # Step 2: Optional reranking
        if self.enabled and self.reranker and chunks:
            logger.info("Reranking %d chunks with top_k=%d", len(chunks), self.top_k)
            chunks = self.reranker.rerank(
                query=query,
                chunks=chunks,
                top_k=self.top_k,
            )
            logger.info("Reranking complete | final chunk count=%d", len(chunks))
        elif self.enabled and not chunks:
            logger.warning("Reranking enabled but no chunks retrieved")
        else:
            logger.info("Reranking skipped")

        # Step 3: Build prompt
        prompt_input = self._build_prompt(chunks)
        logger.info("Prompt built | length=%d characters", len(prompt_input))

        return {
            "query": query,
            "chunks": chunks,
            "prompt_input": prompt_input,
        }

    @staticmethod
    def _build_prompt(chunks: List[Dict]) -> str:
        return "\n\n".join(c["text"] for c in chunks)
