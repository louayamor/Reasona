# Reasona/pipeline/main_pipeline.py

from Reasona.config.config_manager import ConfigurationManager
from Reasona.pipeline.preprocess_pipeline import PreprocessPipeline
from Reasona.pipeline.indexing_pipeline import IndexingPipeline
from Reasona.pipeline.retrieval_pipeline import RetrievalPipeline
from Reasona.pipeline.reranking_pipeline import RerankingPipeline
from Reasona.pipeline.inference_pipeline import InferencePipeline
from Reasona.utils.logger import setup_logger
import torch

logger = setup_logger("main", "logs/pipeline/main_pipeline.json")


def main():
    cfg = ConfigurationManager()

    # ---------------------------
    # Preprocessing & Indexing
    # ---------------------------
    logger.info("Running preprocessing + indexing...")
    preprocess = PreprocessPipeline(cfg.get_preprocess_config())
    data_stream = preprocess.stream()

    indexer = IndexingPipeline(cfg.get_indexing_config())
    indexer.run(data_stream)

    del indexer
    del preprocess
    torch.cuda.empty_cache()

    # ---------------------------
    # Retrieval + Reranking
    # ---------------------------
    retrieval_pipeline = RetrievalPipeline(cfg.get_retrieval_config())
    reranking_pipeline = RerankingPipeline(retrieval_pipeline, cfg)
    retrieval_pipeline.reranking_pipeline = reranking_pipeline

    # ---------------------------
    # Inference pipeline (RAG)
    # ---------------------------
    inference_pipeline = InferencePipeline(
        reranking_pipeline=reranking_pipeline,
        cfg_manager=cfg,
    )
    retrieval_pipeline.inference_pipeline = inference_pipeline

    logger.info("Retrieval + RAG ready. Type a query (or 'exit').")

    # ---------------------------
    # Interactive loop
    # ---------------------------
    while True:
        query = input(">> ")
        if query.lower() in ("exit", "quit"):
            break
        try:
            result = inference_pipeline.run_query(query)
            print("\n=== Generated Article ===\n")
            print(result["answer"])
            print("\n=== Sources / Chunks ===\n")
            for idx, chunk in enumerate(result["chunks"]):
                print(
                    f"[{idx}] score={chunk['score']:.4f} | source={chunk.get('source','unknown')}"
                )
        except Exception as e:
            logger.exception("Error during inference: %s", e)


if __name__ == "__main__":
    main()
