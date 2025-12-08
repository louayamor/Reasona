from pathlib import Path
from Reasona.utils.helpers import read_yaml, create_directories
from Reasona.entities.config_entity import (
    PreprocessConfig,
    TrainingConfig,
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

        create_directories([self.config.get("artifacts_root", "artifacts")])

    # -------------------------------------------------------------
    # PREPROCESS STAGE
    # -------------------------------------------------------------
    def get_preprocess_config(self) -> PreprocessConfig:
        cfg = self.config["preprocess"]

        create_directories([
            cfg["raw_dir"],
            cfg["combined_dir"],
            cfg["processed_dir"],
            cfg["merged_dir"]
        ])

        return PreprocessConfig(
            raw_dir=Path(cfg["raw_dir"]),
            combined_dir=Path(cfg["combined_dir"]),
            processed_dir=Path(cfg["processed_dir"]),
            merged_dir=Path(cfg["merged_dir"]),
            limit=cfg.get("limit", None),
        )

    # -------------------------------------------------------------
    # TRAINING STAGE
    # -------------------------------------------------------------
    def get_training_config(self) -> TrainingConfig:
        cfg = self.config["training"]
        params = self.params["lora"]

        create_directories([cfg["output_dir"]])

        return TrainingConfig(
            dataset_path=Path(cfg["dataset_path"]),
            output_dir=Path(cfg["output_dir"]),
            base_model=cfg["base_model"],
            lora_r=params["r"],
            lora_alpha=params["alpha"],
            lora_dropout=params["dropout"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            learning_rate=params["learning_rate"],
        )

    # -------------------------------------------------------------
    # INFERENCE STAGE
    # -------------------------------------------------------------
    def get_inference_config(self) -> InferenceConfig:
        cfg = self.config["inference"]

        return InferenceConfig(
            model_path=Path(cfg["model_path"]),
            tokenizer_path=Path(cfg["tokenizer_path"]) if cfg["tokenizer_path"] else None,
            inference_engine=cfg["engine"],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )
