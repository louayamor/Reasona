from pathlib import Path
from typing import Dict
from Reasona.pipeline.reranking_pipeline import RerankingPipeline
from Reasona.inference.generator import Generator
from Reasona.config.config_manager import ConfigurationManager
from Reasona.utils.logger import setup_logger

logger = setup_logger(
    "inference_pipeline",
    "logs/pipeline/inference_pipeline.json",
)


class InferencePipeline:
    """
    End-to-end RAG inference pipeline:
    Query → Retrieval → Reranking → 4-bit HF Generation
    """

    def __init__(
        self,
        reranking_pipeline: RerankingPipeline,
        cfg_manager: ConfigurationManager,
    ):
        self.reranking_pipeline = reranking_pipeline

        cfg = cfg_manager.get_inference_config()

        # Initialize generator (supports 4-bit quantization)
        self.generator = Generator(cfg.generator)

        # Load prompt template
        self.prompt_template = self._load_prompt(cfg.prompt.template_path)

        logger.info(
            "Inference initialized | engine=%s | model=%s | 4-bit=%s",
            cfg.engine,
            cfg.generator.model,
            cfg.generator.load_in_4bit,
        )

    def run_query(self, query_text: str) -> Dict:
        """
        End-to-end RAG call:
        1. Retrieval + reranking
        2. Generate final answer with 4-bit HF model
        """

        # Step 1: retrieval + reranking
        reranked = self.reranking_pipeline.run_query(query_text)

        # Step 2: fill prompt template
        prompt = self.prompt_template.format(
            context=reranked["prompt_input"],
            question=query_text,
        )

        # Step 3: generate answer using HF generator
        answer = self.generator.generate(prompt)

        return {
            "query": query_text,
            "answer": answer,
            "chunks": reranked["chunks"],
            "prompt": prompt,
        }

    @staticmethod
    def _load_prompt(path: str) -> str:
        path = str(path) if isinstance(path, (Path, str)) else path
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
