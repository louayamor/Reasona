from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# -----------------------
# Preprocess
# -----------------------
@dataclass(frozen=True)
class PreprocessConfig:
    dataset_name: str
    dataset_config: str
    split: str = "train"
    cache_dir: Optional[Path] = None
    shuffle_buffer: int = 0
    max_samples: Optional[int] = None
    schema_path: Optional[Path] = None


# -----------------------
# Indexing
# -----------------------
@dataclass(frozen=True)
class IndexingConfig:
    vector_store_dir: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    queue_size: int
    max_vectors: Optional[int]
    log_every: Optional[int]
    save_every: Optional[int]
    device: str
    embedding_dim: int


# -----------------------
# Retrieval
# -----------------------
@dataclass(frozen=True)
class RetrievalConfig:
    vector_store_dir: Path
    embedding_model: str
    embedding_dim: int
    top_k: int
    engine: str
    use_cache: bool
    max_workers: int
    debug: bool
    batch_size: int
    log_every: Optional[int]
    device: str
    nprobe: int


# -----------------------
# Reranking
# -----------------------
@dataclass(frozen=True)
class RerankingConfig:
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 10
    batch_size: int = 16


# -----------------------
# Generator (HF / 4-bit aware)
# -----------------------
@dataclass(frozen=True)
class GeneratorConfig:
    provider: str
    model: str
    # generation
    max_tokens: int
    temperature: float
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    revision: Optional[str] = None
    # quantization (HF only)
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "float16"
    double_quant: bool = True
    device_map: str = "auto"


# -----------------------
# Prompt
# -----------------------
@dataclass(frozen=True)
class PromptConfig:
    template_path: Path


# -----------------------
# Inference (RAG)
# -----------------------
@dataclass(frozen=True)
class InferenceConfig:
    engine: str
    embedding_model: str
    generator: GeneratorConfig
    prompt: PromptConfig
