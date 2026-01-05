from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Reasona/entities/config_entity.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class PreprocessConfig:
    dataset_name: str
    split: str
    revision: Optional[str]
    cache_dir: Optional[str]

    max_samples: Optional[int]

    shuffle_buffer: Optional[int]
    prefetch_buffer: Optional[int]

    language: Optional[str] = "en"




@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path                     
    output_dir: Path                       
    base_model: str                        

@dataclass
class IndexingConfig:
    vector_store_dir: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    queue_size: int
    log_every: Optional[int] = 50_000     
    save_every: Optional[int] = 100_000 
    keep_versions: int = 5


@dataclass(frozen=True)
class RetrievalConfig:
    vector_store_dir: Path                 # directory with FAISS / vector store
    top_k: int = 5                         # number of results to retrieve
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    engine: str = "vector_search"          # retrieval engine type


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path                        # path to trained/generation model
    tokenizer_path: Optional[Path] = None   # optional tokenizer path
    engine: str = "transformer"             # inference engine type
    max_tokens: int = 256
    temperature: float = 0.7