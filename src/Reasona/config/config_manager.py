from pathlib import Path
from Reasona.utils.helpers import read_yaml
from typing import Optional
from Reasona.entities.config_entity import (
    PreprocessConfig,
    TrainingConfig,
    IndexingConfig,
    InferenceConfig,
    RetrievalConfig
)
from Reasona.config.validators import require

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_preprocess_config(self) -> PreprocessConfig:
        cfg = self.config.get("preprocess", {})

        def _to_int(value: Optional[int | str], default: Optional[int] = None) -> Optional[int]:
            if value is None:
                return default
            return int(value)

        return PreprocessConfig(
            dataset_name=cfg["dataset_name"],
            dataset_config=cfg.get("dataset_config"),
            split=cfg.get("split", "train"),
            shuffle_buffer=_to_int(cfg.get("shuffle_buffer"), default=0),
            max_samples=_to_int(cfg.get("max_samples")),
            cache_dir=Path(cfg["cache_dir"]).expanduser() if cfg.get("cache_dir") else None,
            schema_path=Path(cfg["schema_path"]) if cfg.get("schema_path") else None,
        )

    def get_indexing_config(self) -> IndexingConfig:
        cfg = self.config.get("indexing")
        if not cfg:
            raise ValueError("Missing 'indexing' section in config.yaml")

        return IndexingConfig(
            vector_store_dir=require(cfg, "vector_store_dir", "indexing"),
            embedding_model=require(cfg, "embedding_model", "indexing"),
            chunk_size=int(require(cfg, "chunk_size", "indexing")),
            chunk_overlap=int(require(cfg, "chunk_overlap", "indexing")),
            batch_size=int(require(cfg, "batch_size", "indexing")),
            queue_size=int(require(cfg, "queue_size", "indexing")),
            log_every=int(cfg.get("log_every")),
            save_every=int(cfg.get("save_every")),
            keep_versions=int(cfg.get("keep_versions")),
        )
        

    def get_retrieval_config(self) -> RetrievalConfig:
        cfg = self.config.get("retrieval")
        if not cfg:
            raise ValueError("Missing 'retrieval' section in config.yaml")

        return RetrievalConfig(
            vector_store_dir=Path(require(cfg, "vector_store_dir", "retrieval")),
            top_k=int(require(cfg, "top_k", "retrieval")),
            embedding_model=require(cfg, "embedding_model", "retrieval"),
            engine=require(cfg, "engine", "retrieval"),
        )

    def get_training_config(self) -> TrainingConfig:
        cfg = self.config.get("training")
        if not cfg:
            raise ValueError("Missing 'training' section in config.yaml")

        return TrainingConfig(
            dataset_path=Path(require(cfg, "dataset_path", "training")),
            output_dir=Path(require(cfg, "output_dir", "training")),
            base_model=require(cfg, "base_model", "training"),
        )

    
    def get_inference_config(self) -> InferenceConfig:
        cfg = self.config.get("inference")
        if not cfg:
            raise ValueError("Missing 'inference' section in config.yaml")

        tokenizer_path = Path(cfg["tokenizer_path"]) if cfg.get("tokenizer_path") else None

        return InferenceConfig(
            model_path=Path(require(cfg, "model_path", "inference")),
            tokenizer_path=tokenizer_path,
            engine=cfg.get("engine", "transformer"),
            max_tokens=int(cfg.get("max_tokens", 512)),
            temperature=float(cfg.get("temperature", 0.7)),
        )
