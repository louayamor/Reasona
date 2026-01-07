from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Reasona/entities/config_entity.py

from dataclasses import dataclass
from typing import Optional

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class PreprocessConfig:
    dataset_name: str                    
    dataset_config: Optional[str] = None 
    split: str = "train"                
    cache_dir: Optional[Path] = None     
    shuffle_buffer: int = 0              
    max_samples: Optional[int] = None    
    schema_path: Optional[Path] = None   

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