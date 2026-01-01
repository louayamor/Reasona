from Reasona.config.config_manager import ConfigurationManager
from Reasona.pipeline.indexing_pipeline import IndexingPipeline
from Reasona.utils.logger import setup_logger

logger = setup_logger("main_pipeline", "logs/pipeline/main_pipeline.log")

def main():
    logger.info("===== PIPELINE STARTED =====")

    try:
        cfg = ConfigurationManager()

        preprocess_cfg = cfg.get_preprocess_config()
        indexing_cfg = cfg.get_indexing_config()

        indexing_pipeline = IndexingPipeline(
            preprocess_cfg=preprocess_cfg,
            indexing_cfg=indexing_cfg,
        )

        indexing_pipeline.run()

        logger.info("===== PIPELINE COMPLETED SUCCESSFULLY =====")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
