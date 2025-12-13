from dataclasses import dataclass
from pathlib import Path

@dataclass
class PreprocessConfig:
    raw_dir: Path
    combined_dir: Path
    processed_dir: Path
    merged_dir: Path
    limit: int | None = None

@dataclass
class TrainingConfig:
    dataset_path: Path
    output_dir: Path
    base_model: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    epochs: int
    learning_rate: float

@dataclass
class InferenceConfig:
    model_path: Path
    tokenizer_path: Path | None
    inference_engine: str     
    max_tokens: int
    temperature: float
