from Reasona.config.config_manager import ConfigurationManager
from Reasona.pipeline.preprocess_pipeline import PreprocessPipeline
from Reasona.pipeline.indexing_pipeline import IndexingPipeline
from Reasona.utils.logger import setup_logger

logger = setup_logger("main", "logs/pipeline/main_pipeline.json")


def main():
    cfg = ConfigurationManager()

    preprocess_cfg = cfg.get_preprocess_config()
    indexing_cfg = cfg.get_indexing_config()

    preprocess_pipeline = PreprocessPipeline(preprocess_cfg)
    indexing_pipeline = IndexingPipeline(indexing_cfg)
    indexing_pipeline.start()

    total_samples = 0
    try:
        for sample in preprocess_pipeline.stream():
            indexing_pipeline.index_chunks(sample)
            total_samples += 1
    finally:
        indexing_pipeline.stop()
        logger.info(f"Indexing finished. Total samples processed: {total_samples}")


if __name__ == "__main__":
    main()
