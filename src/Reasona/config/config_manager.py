from pathlib import Path
from typing import Optional

from Reasona.utils.helpers import read_yaml
from Reasona.entities.config_entity import (
    PreprocessConfig,
    IndexingConfig,
    RetrievalConfig,
    InferenceConfig,
    RerankingConfig,
    GeneratorConfig,
    PromptConfig,
)
from Reasona.config.validators import require

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    def get_preprocess_config(self) -> PreprocessConfig:
        cfg = self.config.get("preprocess", {})

        return PreprocessConfig(
            dataset_name=cfg["dataset_name"],
            dataset_config=cfg["dataset_config"],
            split=cfg.get("split", "train"),
            shuffle_buffer=int(cfg["shuffle_buffer"]),
            max_samples=cfg.get("max_samples"),
            cache_dir=Path(cfg["cache_dir"]).expanduser()
            if cfg.get("cache_dir")
            else None,
            schema_path=Path(cfg["schema_path"])
            if cfg.get("schema_path")
            else None,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def get_indexing_config(self) -> IndexingConfig:
        cfg = self.config.get("indexing")
        if not cfg:
            raise ValueError("Missing 'indexing' section in config.yaml")

        return IndexingConfig(
            vector_store_dir=Path(require(cfg, "vector_store_dir", "indexing")),
            embedding_model=require(cfg, "embedding_model", "indexing"),
            chunk_size=int(require(cfg, "chunk_size", "indexing")),
            chunk_overlap=int(require(cfg, "chunk_overlap", "indexing")),
            batch_size=int(require(cfg, "batch_size", "indexing")),
            queue_size=int(require(cfg, "queue_size", "indexing")),
            max_vectors=int(cfg["max_vectors"])
            if cfg.get("max_vectors") is not None
            else None,
            log_every=int(cfg["log_every"])
            if cfg.get("log_every") is not None
            else None,
            save_every=int(cfg["save_every"])
            if cfg.get("save_every") is not None
            else None,
            device=cfg.get("device", "cuda"),
            embedding_dim=int(cfg.get("embedding_dim", 384)),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def get_retrieval_config(self) -> RetrievalConfig:
        cfg = self.config.get("retrieval")
        if not cfg:
            raise ValueError("Missing 'retrieval' section in config.yaml")

        return RetrievalConfig(
            vector_store_dir=Path(require(cfg, "vector_store_dir", "retrieval")),
            embedding_model=require(cfg, "embedding_model", "retrieval"),
            embedding_dim=int(require(cfg, "embedding_dim", "retrieval")),
            top_k=int(require(cfg, "top_k", "retrieval")),
            engine=require(cfg, "engine", "retrieval"),
            use_cache=bool(cfg.get("use_cache", True)),
            max_workers=int(cfg.get("max_workers", 4)),
            debug=bool(cfg.get("debug", False)),
            batch_size=int(cfg.get("batch_size", 16)),
            log_every=int(cfg["log_every"])
            if cfg.get("log_every") is not None
            else None,
            device=cfg.get("device", "cuda"),
            nprobe=int(cfg.get("nprobe", 64)),
        )

    # ------------------------------------------------------------------
    # Inference (FIXED)
    # ------------------------------------------------------------------
    def get_inference_config(self) -> InferenceConfig:
        cfg = self.config.get("inference")
        if not cfg:
            raise ValueError("Missing 'inference' section in config.yaml")

        gen_cfg = cfg.get("generator")
        if not gen_cfg:
            raise ValueError("Missing 'generator' section under 'inference'")

        prompt_cfg = cfg.get("prompt")
        if not prompt_cfg:
            raise ValueError("Missing 'prompt' section under 'inference'")

        return InferenceConfig(
            engine=require(cfg, "engine", "inference"),
            embedding_model=require(cfg, "embedding_model", "inference"),

            generator=GeneratorConfig(
                provider=require(gen_cfg, "provider", "inference.generator"),
                model=require(gen_cfg, "model", "inference.generator"),
                revision=gen_cfg.get("revision", None),

                # generation
                max_tokens=int(gen_cfg.get("max_tokens", 256)),
                temperature=float(gen_cfg.get("temperature", 0.3)),
                top_p=float(gen_cfg.get("top_p", 0.9)),
                repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.05)),

                # quantization hf 4-bit
                load_in_4bit=bool(gen_cfg.get("load_in_4bit")),
                quant_type=gen_cfg.get("quant_type", "nf4"),
                compute_dtype=gen_cfg.get("compute_dtype", "float16"),
                double_quant=bool(gen_cfg.get("double_quant", True)),
                device_map=gen_cfg.get("device_map", "auto"),
                
            ),

            prompt=PromptConfig(
                template_path=Path(
                    require(prompt_cfg, "template_path", "inference.prompt")
                )
            ),
        )


    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    def get_reranking_config(self) -> RerankingConfig:
        cfg = self.config.get("reranker", {})

        return RerankingConfig(
            enabled=bool(cfg.get("enabled", False)),
            model=cfg.get(
                "model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            top_k=int(cfg.get("top_k", 10)),
            batch_size=int(cfg.get("batch_size", 16)),
        )
