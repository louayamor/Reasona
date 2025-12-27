from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessConfig:
    raw_dir: Path
    combined_dir: Path
    processed_dir: Path
    merged_dir: Path
    output_file: Path
    limit: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    transformed_data_path: Path
    output_dir: Path
    base_model: str


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path
    tokenizer_path: Path | None
    inference_engine: str
    max_tokens: int
    temperature: float
