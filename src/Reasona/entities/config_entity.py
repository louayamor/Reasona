from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class PreprocessConfig:
    dataset_name: str
    split: str = "train"
    revision: str = "main"
    max_samples: Optional[int] = None
    cache_dir: Optional[Path] = Path.home() / ".cache/huggingface/datasets"
    
    # optimize streaming 
    buffer_size: int = 1000         
    prefetch_buffer: int = 500               
    block_size: str = "128MiB"               
    num_workers: int = 2



@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path                     
    output_dir: Path                       
    base_model: str                        

@dataclass(frozen=True)
class IndexingConfig:
    dataset_path: Optional[Path]
    vector_store_dir: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    workers: int
    batch_size: int
    queue_size: int


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