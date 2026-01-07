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
    language: Optional[str] 
    schema_path: Optional[Path]

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
    log_every: Optional[int]    
    save_every: Optional[int]
    keep_versions: int = 5


@dataclass(frozen=True)
class RetrievalConfig:
    vector_store_dir: Path                
    top_k: int                        
    embedding_model: str 
    engine: str         


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path                        
    tokenizer_path: Optional[Path] = None   
    engine: str = "transformer"            
    max_tokens: int = 256
    temperature: float = 0.7