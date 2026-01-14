from pathlib import Path
from typing import Dict, Optional
import time
import uuid

from Reasona.pipeline.reranking_pipeline import RerankingPipeline
from Reasona.inference.generator import Generator
from Reasona.config.config_manager import ConfigurationManager
from Reasona.utils.logger import setup_logger
from Reasona.observability.mlflow_manager import MLflowManager

logger = setup_logger(
    "inference_pipeline",
    "logs/pipeline/inference_pipeline.json",
)


class InferencePipeline:
    """
    End-to-end RAG inference pipeline.
    Retrieval → Reranking → Generation
    Stateless & API-friendly
    """

    def __init__(
        self,
        reranking_pipeline: RerankingPipeline,
        cfg_manager: ConfigurationManager,
        repo_owner: str = "louayamor",
        repo_name: str = "Reasona",
        experiment_name: str = "reasona-inference",
    ):
        logger.info("Initializing InferencePipeline...")
        self.reranking_pipeline = reranking_pipeline

        cfg = cfg_manager.get_inference_config()

        logger.info("Loading Generator model: %s", cfg.generator.model)
        self.generator = Generator(cfg.generator)
        self.prompt_template = self._load_prompt(cfg.prompt.template_path)
        logger.info("Prompt template loaded from %s", cfg.prompt.template_path)

        self.engine = cfg.engine
        self.generator_cfg = cfg.generator

        self.mlflow = MLflowManager(
            repo_owner=repo_owner,
            repo_name=repo_name,
            experiment_name=experiment_name,
        )

        logger.info(
            "InferencePipeline initialized | engine=%s | model=%s | 4bit=%s",
            self.engine,
            self.generator_cfg.model,
            self.generator_cfg.load_in_4bit,
        )

    def execute(self, query: str) -> Dict[str, object]:
        """
        Execute full RAG inference.

        Returns:
        {
            query: str
            answer: str
            chunks: List[Dict]
            prompt: str
            stats: Dict
        }
        """
        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        run_name = f"rag-{uuid.uuid4().hex[:8]}"
        logger.info("Starting inference run: %s for query: '%s'", run_name, query)

        with self.mlflow.run(
            run_name=run_name,
            tags={"pipeline": "rag", "engine": self.engine},
            config=self.generator_cfg,
        ):
            start_time = time.time()

            t0 = time.time()
            logger.info("Executing RerankingPipeline...")
            reranked = self.reranking_pipeline.execute(query)
            latency_rerank = time.time() - t0
            logger.info(
                "Reranking completed | retrieved %d chunks | latency=%.3fs",
                len(reranked["chunks"]),
                latency_rerank,
            )

            prompt = self.prompt_template.format(
                context=reranked["prompt_input"],
                question=query,
            )
            logger.info("Prompt built | length=%d characters", len(prompt))
            self._log_artifact(
                content=prompt,
                artifact_dir="prompts",
                filename=f"{run_name}_prompt.txt",
            )

            t1 = time.time()
            logger.info("Generating answer with Generator model...")
            answer = self.generator.generate(prompt)
            latency_generation = time.time() - t1
            logger.info("Answer generation completed | latency=%.3fs", latency_generation)

            self._log_artifact(
                content=answer,
                artifact_dir="answers",
                filename=f"{run_name}_answer.txt",
            )

            latency_total = time.time() - start_time

            self.mlflow.log_params({
                "query": query,
                "num_chunks": len(reranked["chunks"]),
            })
            self.mlflow.log_metrics({
                "latency_rerank": latency_rerank,
                "latency_generation": latency_generation,
                "latency_total": latency_total,
            })

            logger.info(
                "Inference run completed | run=%s | total latency=%.3fs | chunks=%d",
                run_name,
                latency_total,
                len(reranked["chunks"]),
            )

            return {
                "query": query,
                "answer": answer,
                "chunks": reranked["chunks"],
                "prompt": prompt,
                "stats": {
                    "latency_rerank": latency_rerank,
                    "latency_generation": latency_generation,
                    "latency_total": latency_total,
                },
            }

    def _log_artifact(self, content: str, artifact_dir: str, filename: str) -> None:
        path = Path(f"artifacts/{artifact_dir}/{filename}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.mlflow.log_artifact(str(path), artifact_path=artifact_dir)
        logger.info("Artifact saved: %s", path)

    @staticmethod
    def _load_prompt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info("Loaded prompt template from %s", path)
        return content
