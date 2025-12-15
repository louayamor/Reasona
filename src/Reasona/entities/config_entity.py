from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class PreprocessConfig:
    raw_dir: Path
    combined_dir: Path
    processed_dir: Path
    merged_dir: Path
    limit: Optional[int] = None

@dataclass(frozen=True)
class TrainingConfig:
    transformed_data_path: Path
    output_dir: Path

    base_model: Optional[str] = None

    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None

    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None



@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path
    tokenizer_path: Optional[Path] = None

    inference_engine: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
