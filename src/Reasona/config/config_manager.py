from pathlib import Path
from Reasona.utils.helpers import read_yaml
from Reasona.entities.config_entity import (
    PreprocessConfig,
    TrainingConfig,
    IndexingConfig,
    InferenceConfig,
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
        cfg = self.config.get("preprocess")
        if not cfg:
            raise ValueError("Missing 'preprocess' section in config.yaml")
        
        cache_dir = Path(cfg["cache_dir"]) if cfg.get("cache_dir") else Path.home() / ".cache/huggingface/datasets"

        return PreprocessConfig(
        dataset_name=cfg.get("dataset_name", "PleIAs/SYNTH"),
        split=cfg.get("split", "train"),
        max_samples=cfg.get("max_samples"),
        revision=cfg.get("revision", "main"),
        cache_dir=cache_dir,
        buffer_size=int(cfg.get("buffer_size", 1000)),
        prefetch_buffer=int(cfg.get("prefetch_buffer", 500)),
        block_size=cfg.get("block_size", "128MiB"),
        num_workers=int(cfg.get("num_workers", 2)),
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

    def get_indexing_config(self) -> IndexingConfig:
        cfg = self.config.get("indexing")
        if not cfg:
            raise ValueError("Missing 'indexing' section in config.yaml")

        dataset_path_raw = cfg.get("dataset_path")
        dataset_path = Path(dataset_path_raw) if dataset_path_raw else None

        return IndexingConfig(
            dataset_path=dataset_path,
            vector_store_dir=Path(cfg.get("vector_store_dir", "artifacts/vectors")),
            embedding_model=cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            chunk_size=int(cfg.get("chunk_size", 256)),
            chunk_overlap=int(cfg.get("chunk_overlap", 32)),
            workers=int(cfg.get("workers", 2)),
            batch_size=int(cfg.get("batch_size", 64)),
            queue_size=int(cfg.get("queue_size", 512)),
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
