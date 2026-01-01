from pathlib import Path

from Reasona.config.config_manager import ConfigurationManager
from Reasona.pipeline.preprocess_pipeline import PreprocessPipeline
from Reasona.pipeline.indexing_pipeline import IndexingPipeline
from Reasona.data.chunker import TextChunker
from Reasona.data.embedder import Embedder


def main():
    # Load configs from ConfigurationManager
    cfg_manager = ConfigurationManager()
    preprocess_cfg = cfg_manager.get_preprocess_config()
    indexing_cfg = cfg_manager.get_indexing_config()

    preprocess_pipeline = PreprocessPipeline(preprocess_cfg)

    chunker = TextChunker(
        chunk_size=indexing_cfg.chunk_size,
        chunk_overlap=indexing_cfg.chunk_overlap,
    )
    embedder = Embedder(model_name=indexing_cfg.embedding_model)

    indexing_pipeline = IndexingPipeline(
        preprocess_pipeline=preprocess_pipeline,
        chunker=chunker,
        embedder=embedder,
        vector_db_dir=indexing_cfg.vector_store_dir,
        workers=4,  
    )

    indexing_pipeline.run()


if __name__ == "__main__":
    main()
