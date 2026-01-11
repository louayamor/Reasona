from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class PreprocessConfig:
    dataset_name: str                    
    dataset_config: str 
    split: str = "train"                
    cache_dir: Optional[Path] = None     
    shuffle_buffer: int = 0              
    max_samples: Optional[int] = None    
    schema_path: Optional[Path] = None                       

@dataclass
class IndexingConfig:
    vector_store_dir: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    queue_size: int
    log_every: Optional[int]    
    save_every: Optional[int]
    max_vectors: Optional[int] 
    device: str 
    embedding_dim: int

@dataclass(frozen=True)
class RetrievalConfig:
    vector_store_dir: Path
    top_k: int
    embedding_model: str
    engine: str
    embedding_dim: int
    use_cache: bool
    max_workers: int
    debug: bool
    batch_size: int
    log_every: Optional[int]
    device: str 
    nprobe: int 

@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path                        
    tokenizer_path: Optional[Path] = None   
    engine: str = "transformer"            
    max_tokens: int = 256
    temperature: float = 0.7

@dataclass(frozen=True)
class RerankingConfig:
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 10
    batch_size: int = 16