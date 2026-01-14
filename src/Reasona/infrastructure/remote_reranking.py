from typing import Dict, List, Optional
import shutil
import tempfile
import os

from Reasona.inference.reranker import Reranker
from Reasona.config.config_manager import ConfigurationManager
from Reasona.utils.logger import setup_logger
from Reasona.infrastructure.remote_retrieval import RemoteRetrievalPipeline

logger = setup_logger(
    "remote_reranking_pipeline",
    "logs/pipeline/remote_reranking.json",
)

class RemoteRerankingPipeline:
    """
    Stateless pipeline for remote retrieval + optional reranking.
    - Downloads shards into /tmp
    - Loads ShardedFaissStore for retrieval
    - Optionally reranks retrieved chunks
    - Cleans temp after each run
    """

    def __init__(self, cfg_manager: ConfigurationManager):
        logger.info("Initializing RemoteRerankingPipeline...")

        retrieval_cfg = cfg_manager.get_retrieval_config()
        self.retrieval_pipeline = RemoteRetrievalPipeline(retrieval_cfg)

        cfg = cfg_manager.get_reranking_config()
        self.enabled: bool = cfg.enabled
        self.top_k: int = cfg.top_k
        self.reranker: Optional[Reranker] = None

        if self.enabled:
            logger.info("Loading Reranker model: %s", cfg.model)
            self.reranker = Reranker(
                model_name=cfg.model,
                batch_size=cfg.batch_size,
                device=getattr(cfg, "device", "cuda")
            )
            logger.info("Reranker loaded successfully")
        else:
            logger.info("Reranking disabled")

        logger.info(
            "RemoteRerankingPipeline initialized | reranking_enabled=%s | top_k=%d",
            self.enabled,
            self.top_k,
        )

    def execute(
        self,
        query: str,
        retrieval_top_k: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Execute retrieval + optional reranking.
        Returns:
        {
            "query": str,
            "chunks": List[Dict],
            "prompt_input": str,
        }
        """
        logger.info("Executing RemoteRerankingPipeline for query: '%s'", query)

        # Step 1: Retrieve chunks remotely
        retrieval_result = self.retrieval_pipeline.execute(
            query=query,
            top_k=retrieval_top_k
        )
        chunks: List[Dict] = retrieval_result["chunks"]
        logger.info(
            "Retrieved %d chunks | retrieval_top_k=%s",
            len(chunks),
            retrieval_top_k or self.retrieval_pipeline.cfg.top_k,
        )

        if self.enabled and self.reranker and chunks:
            logger.info("Reranking %d chunks | top_k=%d", len(chunks), self.top_k)
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

        prompt_input = self._build_prompt(chunks)
        logger.info("Prompt built | length=%d characters", len(prompt_input))

        temp_dir = self.retrieval_pipeline.temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Temp directory cleared: %s", temp_dir)

        return {
            "query": query,
            "chunks": chunks,
            "prompt_input": prompt_input,
        }

    @staticmethod
    def _build_prompt(chunks: List[Dict]) -> str:
        return "\n\n".join(c["text"] for c in chunks)
