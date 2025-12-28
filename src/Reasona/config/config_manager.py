from pathlib import Path
from Reasona.utils.helpers import read_yaml, create_directories
from Reasona.entities.config_entity import (
    PreprocessConfig,
    TrainingConfig,
    EmbeddingConfig,
    InferenceConfig,
)

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

    # -----------------------------
    # PREPROCESS
    # -----------------------------
    def get_preprocess_config(self) -> PreprocessConfig:
        cfg = self.config["preprocess"]

        raw_dir = Path(cfg["raw_dir"])
        combined_dir = Path(cfg["combined_dir"])
        processed_dir = Path(cfg["processed_dir"])
        merged_dir = Path(cfg["merged_dir"])
        output_file = Path(cfg["output_file"])

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
            output_file=output_file,
            limit=cfg.get("limit"),
        )

    # -----------------------------
    # TRAINING (optional)
    # -----------------------------
    def get_training_config(self) -> TrainingConfig:
        cfg = self.config["training"]

        output_dir = Path(cfg["output_dir"])
        create_directories([output_dir])

        return TrainingConfig(
            transformed_data_path=Path(cfg["transformed_data_path"]),
            output_dir=output_dir,
            base_model=cfg["base_model"],
        )

    # -----------------------------
    # EMBEDDING (RAG core)
    # -----------------------------
    def get_embedding_config(self) -> EmbeddingConfig:
        cfg = self.config["embedding"]

        vector_db_dir = Path(cfg["vector_db_dir"])
        create_directories([vector_db_dir])

        return EmbeddingConfig(
            dataset_path=Path(cfg["dataset_path"]),
            vector_db_dir=vector_db_dir,
            embedding_model=cfg["embedding_model"],
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )

    # -----------------------------
    # INFERENCE
    # -----------------------------
    def get_inference_config(self) -> InferenceConfig:
        cfg = self.config["inference"]

        tokenizer_path = (
            Path(cfg["tokenizer_path"])
            if cfg.get("tokenizer_path")
            else None
        )

        return InferenceConfig(
            model_path=Path(cfg["model_path"]),
            tokenizer_path=tokenizer_path,
            inference_engine=cfg["engine"],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )
