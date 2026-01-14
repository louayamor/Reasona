import time
from pathlib import Path
from typing import Dict, Any, Iterable, List
import boto3
import os
import numpy as np
import shutil
import tempfile

from Reasona.data.embedder import Embedder
from Reasona.data.chunker import TextChunker
from Reasona.config.config_manager import IndexingConfig
from Reasona.utils.logger import setup_logger
from Reasona.vectorstore.sharded_faiss_store import ShardedFaissStore  # independent sharded store


class RemoteIndexingPipeline:
    """
    Sharded FAISS indexing pipeline for Backblaze B2.
    - Uses /tmp for Render deployment
    - Max shard size controlled (~400 MB)
    - Uploads each shard to B2 immediately
    """

    def __init__(self, cfg: IndexingConfig):
        self.cfg = cfg
        self.logger = setup_logger(
            "remote_indexing_pipeline",
            "logs/pipeline/remote_indexing.json",
        )

        self.embedder = Embedder(
            model_name=cfg.embedding_model,
            batch_size=cfg.batch_size,
            device=cfg.device,
            log_every=cfg.log_every,
        )

        self.chunker = TextChunker(
            chunk_size=cfg.chunk_size,
            overlap=cfg.chunk_overlap,
            log_every=cfg.log_every,
        )

        # Temporary directory for Render
        self.temp_dir = Path(tempfile.mkdtemp())
        self.logger.info("Temp directory created: %s", self.temp_dir)

        # Sharded FAISS store in /tmp
        self.store = ShardedFaissStore(
            dim=cfg.embedding_dim,
            base_path=self.temp_dir,
            max_vectors_per_shard=cfg.max_vectors_per_shard or 50_000,  # ~400MB per shard estimate
            nprobe=getattr(cfg, "nprobe", 16),
            mmap=False,
        )

        self.vectors_written = 0
        self.start_time = time.time()

        # B2 client
        self.s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["B2_ENDPOINT"],
            aws_access_key_id=os.environ["B2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["B2_SECRET_ACCESS_KEY"],
        )
        self.bucket_name = os.environ["B2_BUCKET_NAME"]

        self.logger.info(
            "Remote sharded indexing initialized | temp_dir=%s",
            self.temp_dir,
        )

    def run(self, stream: Iterable[Dict[str, Any]]):
        buffer: List[Dict[str, Any]] = []

        for item in stream:
            for chunk in self.chunker.chunk_item(item):
                buffer.append(chunk)

                if len(buffer) >= self.cfg.batch_size:
                    self._process_batch(buffer)
                    buffer.clear()

        if buffer:
            self._process_batch(buffer)

        self._finalize()

    def _process_batch(self, batch: List[Dict[str, Any]]):
        texts = [c["text"] for c in batch]

        metas = [
            {
                "id": c["id"],
                "text": c["text"],
                "source": c.get("source"),
                "original_id": c.get("original_id"),
            }
            for c in batch
        ]

        vectors = self.embedder.embed(texts)
        added = self.store.add(vectors, metas)
        self.vectors_written += added

        del vectors, metas, texts

        if self.vectors_written % self.cfg.log_every < self.cfg.batch_size:
            self._log_progress()

        if self.vectors_written % self.cfg.save_every < self.cfg.batch_size:
            self._checkpoint()

    def _log_progress(self):
        elapsed = max(time.time() - self.start_time, 1e-6)
        rate = self.vectors_written / elapsed
        self.logger.info("Indexing progress | vectors=%d | rate=%.1f vec/s",
                         self.vectors_written, rate)

    def _checkpoint(self):
        self.store.save()
        self._upload_shards()

    def _upload_shards(self):
        """
        Upload all FAISS shards to B2 and clear local temp after upload.
        """
        for shard_path in sorted(self.temp_dir.glob("shard_*.faiss")):
            db_path = shard_path.with_suffix(".db")
            # Upload index
            self.s3.upload_file(str(shard_path), self.bucket_name, shard_path.name)
            self.logger.info("Uploaded shard %s to B2", shard_path.name)
            # Upload metadata DB
            if db_path.exists():
                self.s3.upload_file(str(db_path), self.bucket_name, db_path.name)
                self.logger.info("Uploaded shard DB %s to B2", db_path.name)
            # Clear temp files
            shard_path.unlink()
            if db_path.exists():
                db_path.unlink()

    def _finalize(self):
        self.store.finalize()
        self._upload_shards()

        # Remove temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.logger.info(
            "Indexing finished | total_vectors=%d | runtime=%.1fs | temp_dir cleared",
            self.vectors_written,
            time.time() - self.start_time,
        )
