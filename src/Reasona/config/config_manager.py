from pathlib import Path
from Reasona.utils.helpers import read_yaml, create_directories
from Reasona.entities.config_entity import PreprocessConfig, TrainingConfig, InferenceConfig

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

    # -------------------------------------------------------------
    # PREPROCESS STAGE
    # -------------------------------------------------------------
    def get_preprocess_config(self) -> PreprocessConfig:
        cfg = self.config["preprocess"]

        dirs = [
            Path(cfg["raw_dir"]),
            Path(cfg["combined_dir"]),
            Path(cfg["processed_dir"]),
            Path(cfg["merged_dir"])
        ]
        create_directories(dirs)

        return PreprocessConfig(
            raw_dir=dirs[0],
            combined_dir=dirs[1],
            processed_dir=dirs[2],
            merged_dir=dirs[3],
            limit=cfg.get("limit", None),
        )

    # -------------------------------------------------------------
    # TRAINING STAGE
    # -------------------------------------------------------------
    def get_training_config(self) -> TrainingConfig:
        cfg = self.config["training"]
        params = self.params.get("lora", {})

        output_dir = Path(cfg["output_dir"])
        create_directories([output_dir])

        return TrainingConfig(
            dataset_path=Path(cfg["dataset_path"]),
            output_dir=output_dir,
            base_model=cfg.get("base_model"),
            lora_r=params.get("r"),
            lora_alpha=params.get("alpha"),
            lora_dropout=params.get("dropout"),
            batch_size=params.get("batch_size"),
            epochs=params.get("epochs"),
            learning_rate=params.get("learning_rate"),
        )

    # -------------------------------------------------------------
    # INFERENCE STAGE
    # -------------------------------------------------------------
    def get_inference_config(self) -> InferenceConfig:
        cfg = self.config["inference"]

        tokenizer_path = Path(cfg["tokenizer_path"]) if cfg.get("tokenizer_path") else None

        return InferenceConfig(
            model_path=Path(cfg["model_path"]),
            tokenizer_path=tokenizer_path,
            inference_engine=cfg.get("engine"),
            max_tokens=cfg.get("max_tokens"),
            temperature=cfg.get("temperature"),
        )
