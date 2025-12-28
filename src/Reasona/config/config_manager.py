from pathlib import Path
from Reasona.utils.helpers import read_yaml, create_directories
from Reasona.entities.config_entity import (
    PreprocessConfig,
    TrainingConfig,
    EmbeddingConfig,
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

        artifacts_root = Path(self.config.get("artifacts_root", "artifacts"))
        create_directories([artifacts_root])

    # ---------- PREPROCESS ----------
    def get_preprocess_config(self) -> PreprocessConfig:
        cfg = self.config.get("preprocess")
        if not cfg:
            raise ValueError("Missing 'preprocess' section in config.yaml")

        raw_dir = Path(require(cfg, "raw_dir", "preprocess"))
        combined_dir = Path(require(cfg, "combined_dir", "preprocess"))
        processed_dir = Path(require(cfg, "processed_dir", "preprocess"))
        merged_dir = Path(require(cfg, "merged_dir", "preprocess"))

        create_directories([
            raw_dir,
            combined_dir,
            processed_dir,
            merged_dir,
        ])

        return PreprocessConfig(
            raw_dir=raw_dir,
            combined_dir=combined_dir,
            processed_dir=processed_dir,
            merged_dir=merged_dir,
            limit=cfg.get("limit"),
        )

    # ---------- TRAINING ----------
    def get_training_config(self) -> TrainingConfig:
        cfg = self.config.get("training")
        if not cfg:
            raise ValueError("Missing 'training' section in config.yaml")

        transformed_data_path = Path(
            require(cfg, "transformed_data_path", "training")
        )
        output_dir = Path(require(cfg, "output_dir", "training"))
        base_model = require(cfg, "base_model", "training")

        create_directories([output_dir])

        return TrainingConfig(
            transformed_data_path=transformed_data_path,
            output_dir=output_dir,
            base_model=base_model,
        )

    # ---------- EMBEDDING ----------
    def get_embedding_config(self) -> EmbeddingConfig:
        cfg = self.config.get("embedding")
        if not cfg:
            raise ValueError("Missing 'embedding' section in config.yaml")

        dataset_path = Path(require(cfg, "dataset_path", "embedding"))
        vector_db_dir = Path(require(cfg, "vector_db_dir", "embedding"))
        embedding_model = require(cfg, "embedding_model", "embedding")

        chunk_size = int(cfg.get("chunk_size", 512))
        chunk_overlap = int(cfg.get("chunk_overlap", 50))

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "embedding.chunk_overlap must be smaller than chunk_size"
            )

        create_directories([vector_db_dir])

        return EmbeddingConfig(
            dataset_path=dataset_path,
            vector_db_dir=vector_db_dir,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ---------- INFERENCE ----------
    def get_inference_config(self) -> InferenceConfig:
        cfg = self.config.get("inference")
        if not cfg:
            raise ValueError("Missing 'inference' section in config.yaml")

        model_path = Path(require(cfg, "model_path", "inference"))

        tokenizer_path = (
            Path(cfg["tokenizer_path"])
            if cfg.get("tokenizer_path")
            else None
        )

        return InferenceConfig(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            inference_engine=cfg.get("engine", "python"),
            max_tokens=int(cfg.get("max_tokens", 256)),
            temperature=float(cfg.get("temperature", 0.7)),
        )
