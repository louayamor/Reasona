from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# -----------------------------
# Streaming / Preprocessing configuration
# -----------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    dataset_name: str
    split: str
    revision: str
    max_samples: Optional[int] = None
    cache_dir: Optional[str] = None



# -----------------------------
# Training configuration
# -----------------------------
@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path                     # path to transformed dataset
    output_dir: Path                       # directory to save trained models
    base_model: str                        # name of base model


# -----------------------------
# Embedding / Indexing configuration
# -----------------------------
@dataclass(frozen=True)
class IndexingConfig:
    dataset_path: Optional[Path] = None    
    vector_store_dir: Path = Path("artifacts/vectors")
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 256
    chunk_overlap: int = 32
    workers: int = 2
    batch_size: int = 64

# -----------------------------
# Retrieval / Inference configuration
# -----------------------------
@dataclass(frozen=True)
class RetrievalConfig:
    vector_store_dir: Path                 # directory with FAISS / vector store
    top_k: int = 5                         # number of results to retrieve
    embedding_model: str = "all-MiniLM-L6-v2"
    engine: str = "vector_search"          # retrieval engine type


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path                        # path to trained/generation model
    tokenizer_path: Optional[Path] = None   # optional tokenizer path
    engine: str = "transformer"             # inference engine type
    max_tokens: int = 512
    temperature: float = 0.7
