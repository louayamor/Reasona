# Reasona/pipeline/inference_pipeline.py

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
    End-to-end inference:
    Query → Retrieval → Reranking → Generation
    """

    def __init__(
        self,
        reranking_pipeline: RerankingPipeline,
        cfg_manager: ConfigurationManager,
    ):
        self.reranking_pipeline = reranking_pipeline

        cfg = cfg_manager.get_inference_config()

        self.generator = Generator(cfg.generator)
        self.prompt_template = self._load_prompt(
            cfg.prompt.template_path
        )

        logger.info(
            "Inference initialized engine=%s model=%s",
            cfg.engine,
            cfg.generator.model,
        )

    def run_query(self, query_text: str) -> Dict:
        """
        Final user-facing inference call
        """

        reranked = self.reranking_pipeline.run_query(query_text)

        prompt = self.prompt_template.format(
            context=reranked["prompt_input"],
            question=query_text,
        )

        answer = self.generator.generate(prompt)

        return {
            "query": query_text,
            "answer": answer,
            "chunks": reranked["chunks"],
            "prompt": prompt,
        }

    @staticmethod
    def _load_prompt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
