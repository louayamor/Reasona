from Reasona.config.config_manager import ConfigurationManager
from Reasona.pipeline.preprocess_pipeline import PreprocessPipeline
from Reasona.pipeline.indexing_pipeline import IndexingPipeline
from Reasona.utils.logger import setup_logger
import torch
import gc

logger = setup_logger("main", "logs/pipeline/main_pipeline.json")

def main():
    cfg = ConfigurationManager()

    preprocess = PreprocessPipeline(cfg.get_preprocess_config())
    indexer = IndexingPipeline(cfg.get_indexing_config())

    indexer.run(preprocess.stream())

    del indexer
    del preprocess

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
