from pathlib import Path
from Reasona.config.config_manager import ConfigurationManager
from Reasona.pipeline.preprocess_pipeline import PreprocessPipeline
from Reasona.pipeline.indexing_pipeline import IndexingPipeline
from Reasona.data.embedder import Embedder

def main():
    cfg = ConfigurationManager()
    preprocess_cfg = cfg.get_preprocess_config()
    indexing_cfg = cfg.get_indexing_config()

    # streaming producer
    preprocess_pipeline = PreprocessPipeline(preprocess_cfg)

    # indexing (consumer)
    embedder = Embedder(indexing_cfg.embedding_model)
    indexing_pipeline = IndexingPipeline(
        embedder=embedder,
        vector_db_dir=indexing_cfg.vector_store_dir,
        workers=indexing_cfg.workers,
        batch_size=indexing_cfg.batch_size,
    )

    indexing_pipeline.start()

    for sample in preprocess_pipeline.stream():
        indexing_pipeline.index_chunks(sample)

    indexing_pipeline.stop()


if __name__ == "__main__":
    main()
