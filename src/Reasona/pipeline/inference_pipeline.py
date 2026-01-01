from typing import List, Dict
from Reasona.utils.logger import setup_logger
from Reasona.pipeline.retrieval_pipeline import RetrievalPipeline
from Reasona.entities.config_entity import InferenceConfig

logger = setup_logger("inference_pipeline", "logs/pipeline/inference_pipeline.json")


class InferencePipeline:
    """
    Inference pipeline using RetrievalPipeline to provide context for prompts.
    """

    def __init__(self, retrieval_pipeline: RetrievalPipeline, cfg: InferenceConfig, model=None):
        self.retrieval = retrieval_pipeline
        self.cfg = cfg
        self.model = model  # Any callable that takes a prompt and returns output

    def generate_prompt(self, user_prompt: str, top_k_contexts: List[Dict]) -> str:
        """
        Build a prompt by combining retrieved context chunks with the user prompt.
        """
        context_text = "\n\n".join([c.get("text", "") for c in top_k_contexts])
        full_prompt = f"Context:\n{context_text}\n\nQuestion:\n{user_prompt}"
        return full_prompt

    def run(self, user_prompt: str, top_k: int = 5) -> str:
        """
        Run retrieval + model inference.
        Returns the generated answer.
        """
        logger.info(f"Running inference for prompt: {user_prompt[:50]}...")

        retrieved_chunks = self.retrieval.query(user_prompt, top_k=top_k)

        final_prompt = self.generate_prompt(user_prompt, retrieved_chunks)

        if self.model is None:
            logger.warning("No model provided. Returning prompt only.")
            return final_prompt

        output = self.model(final_prompt)
        logger.info("Inference completed.")
        return output
