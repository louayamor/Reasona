from pathlib import Path
from typing import Dict
import time
import uuid
import os

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
    End-to-end RAG inference pipeline:
    Query → Retrieval → Reranking → 4-bit HF Generation
    MLflow logging integrated for Dagshub
    """

    def __init__(
        self,
        reranking_pipeline: RerankingPipeline,
        cfg_manager: ConfigurationManager,
    ):
        self.reranking_pipeline = reranking_pipeline

        cfg = cfg_manager.get_inference_config()

        self.generator = Generator(cfg.generator)
        self.prompt_template = self._load_prompt(cfg.prompt.template_path)

        # MLflow manager (Dagshub-compatible)
        self.mlflow = MLflowManager("reasona-inference")

        self.engine = cfg.engine
        self.model_name = cfg.generator.model
        self.load_in_4bit = cfg.generator.load_in_4bit

        logger.info(
            "Inference initialized | engine=%s | model=%s | 4-bit=%s",
            self.engine,
            self.model_name,
            self.load_in_4bit,
        )

    def run_query(self, query_text: str) -> Dict:
        """
        End-to-end RAG call:
        1. Retrieval + reranking
        2. Generate final answer with 4-bit HF model
        MLflow logging for params, metrics, and artifacts
        """

        # Generate unique run name for Dagshub
        run_name = f"rag-inference-{uuid.uuid4().hex[:8]}"

        with self.mlflow.run(
            run_name=run_name,
            tags={
                "pipeline": "rag",
                "engine": self.engine,
            },
        ):
            start_time = time.time()

            # ---- Retrieval + Reranking ----
            t0 = time.time()
            reranked = self.reranking_pipeline.run_query(query_text)
            rerank_time = time.time() - t0

            # ---- Prompt Construction ----
            prompt = self.prompt_template.format(
                context=reranked["prompt_input"],
                question=query_text,
            )

            # Save prompt as artifact
            prompt_path = Path(f"artifacts/prompts/{run_name}_prompt.txt")
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            self.mlflow.log_artifact(str(prompt_path), artifact_path="prompts")

            # ---- Generation ----
            t1 = time.time()
            answer = self.generator.generate(prompt)
            generation_time = time.time() - t1

            # Save answer as artifact
            answer_path = Path(f"artifacts/answers/{run_name}_answer.txt")
            answer_path.parent.mkdir(parents=True, exist_ok=True)
            with open(answer_path, "w", encoding="utf-8") as f:
                f.write(answer)
            self.mlflow.log_artifact(str(answer_path), artifact_path="answers")

            total_time = time.time() - start_time

            # ---- MLflow Logging ----
            self.mlflow.log_params({
                "generator_model": self.model_name,
                "load_in_4bit": self.load_in_4bit,
                "query_text": query_text,
                "num_chunks_retrieved": len(reranked["chunks"]),
            })

            self.mlflow.log_metrics({
                "latency_rerank": rerank_time,
                "latency_generation": generation_time,
                "latency_total": total_time,
            })

            logger.info(
                "Query processed | run=%s | total_time=%.3fs | chunks=%d",
                run_name, total_time, len(reranked["chunks"])
            )

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
