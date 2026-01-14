from pathlib import Path
from typing import List, Dict, Callable, Optional
import tempfile
import shutil
import os
import boto3
import numpy as np

from Reasona.data.embedder import Embedder
from Reasona.vectorstore.sharded_faiss_store import ShardedFaissStore
from Reasona.inference.retriever import Retriever
from Reasona.entities.config_entity import RetrievalConfig
from Reasona.utils.logger import setup_logger

import faiss

faiss.omp_set_num_threads(1)

logger = setup_logger(
    "remote_retrieval_pipeline",
    "logs/pipeline/remote_retrieval.json",
)


class RemoteRetrievalPipeline:
    """
    Remote retrieval pipeline that loads sharded FAISS indexes from Backblaze B2.
    - Downloads shards into /tmp
    - Loads into memory using ShardedFaissStore
    - Cleans temp after retrieval
    """

    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info("Temp directory created: %s", self.temp_dir)

        self.s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["B2_ENDPOINT"],
            aws_access_key_id=os.environ["B2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["B2_SECRET_ACCESS_KEY"],
        )
        self.bucket_name = os.environ["B2_BUCKET_NAME"]

        self._download_shards()

        self.store = ShardedFaissStore(
            dim=cfg.embedding_dim,
            base_path=self.temp_dir,
            max_vectors_per_shard= 50_000,
            nprobe=getattr(cfg, "nprobe", 16),
            mmap=False,
        )

        self.embedder = Embedder(
            model_name=cfg.embedding_model,
            batch_size=1,
            device="cuda",
            log_every=cfg.log_every,
        )

        self.retriever = Retriever()
        logger.info("RemoteRetrievalPipeline initialized with %d shards", len(self.store.shards))

    def _download_shards(self):
        """Download all .faiss and .db files from B2 into temp_dir"""
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".faiss") or key.endswith(".db"):
                    dest = self.temp_dir / key
                    self.s3.download_file(self.bucket_name, key, str(dest))
                    logger.info("Downloaded %s to %s", key, dest)

    def execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_scores: bool = True,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> Dict[str, object]:
        if not query or not query.strip():
            logger.warning("Empty query received")
            raise ValueError("query must be non-empty")

        top_k = top_k or self.cfg.top_k
        logger.info("Top-k set to %d", top_k)

        logger.info("Embedding query text")
        query_vector = self.embedder.embed([query]).astype("float32")

        logger.info("Retrieving top %d chunks from sharded FAISS", top_k)
        chunks = self.retriever.retrieve(
            query_vector=query_vector,
            k=top_k,
            return_scores=return_scores,
            filter_fn=filter_fn,
            index=self.store,
        )
        logger.info("Retrieved %d chunks", len(chunks))

        prompt_input = self._build_prompt(chunks)
        logger.info("Built prompt of length %d characters", len(prompt_input))

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        logger.info("Temp directory cleared: %s", self.temp_dir)

        return {
            "query": query,
            "chunks": chunks,
            "prompt_input": prompt_input,
            "stats": {
                "top_k": top_k,
                "num_chunks": len(chunks),
            },
        }

    @staticmethod
    def _build_prompt(chunks: List[Dict]) -> str:
        return "\n\n".join(c["text"] for c in chunks)

    @staticmethod
    def filter_by_source(source: str) -> Callable[[Dict], bool]:
        return lambda meta: meta.get("source") == source
