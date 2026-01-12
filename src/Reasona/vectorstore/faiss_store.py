from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sqlite3
import numpy as np
import faiss

from Reasona.utils.logger import setup_logger


class FaissStore:
    """
    FAISS vector store with SQLite metadata.
    Supports:
    - WRITE mode (indexing): train + add + save
    - READONLY mode (inference): mmap + search only
    """

    def __init__(
        self,
        dim: Optional[int],
        index_path: Path,
        db_path: Path,
        max_vectors: Optional[int] = None,
        nprobe: int = 16,
        mmap: bool = True,
        readonly: bool = False,
    ):
        self.dim = dim
        self.index_path = index_path
        self.db_path = db_path
        self.max_vectors = max_vectors
        self.nprobe = nprobe
        self.mmap = mmap
        self.readonly = readonly

        self.index: Optional[faiss.Index] = None
        self.conn: Optional[sqlite3.Connection] = None
        self._train_buffer: List[np.ndarray] = []

        self.logger = setup_logger("faiss_store", "logs/vectorstore/faiss_store.json")
        self.logger.info(
            "FaissStore initialized | index=%s | db=%s | readonly=%s | mmap=%s",
            self.index_path,
            self.db_path,
            self.readonly,
            self.mmap,
        )

    # ------------------------------------------------------------------
    # Loading / Saving
    # ------------------------------------------------------------------
    def load(self):
        self._open_db()
        self._ensure_schema()

        if self.index_path.exists():
            flags = faiss.IO_FLAG_MMAP if (self.mmap and self.readonly) else 0
            self.index = faiss.read_index(str(self.index_path), flags)

            if hasattr(self.index, "nprobe"):
                self.index.nprobe = self.nprobe

            # Never retrain or add in readonly mode
            if not self.readonly and hasattr(self.index, "is_trained") and not self.index.is_trained:
                self.logger.info("Index not trained, retraining from DB")
                self._retrain_from_db()

    def save(self):
        if self.readonly:
            return
        if self.index:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))

    def close(self):
        if self.conn:
            self.conn.close()

    def finalize(self):
        self.save()
        self.close()

    # ------------------------------------------------------------------
    # Index creation & training
    # ------------------------------------------------------------------
    def _create_index(self, dim: int, nlist: Optional[int] = None):
        if nlist is None:
            if self.max_vectors:
                nlist = min(4096, max(1, int(np.sqrt(self.max_vectors))))
            else:
                nlist = 1024

        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(
            quantizer, dim, nlist, faiss.METRIC_L2
        )
        self.index.nprobe = self.nprobe

    def _retrain_from_db(self):
        cur = self.conn.cursor()
        cur.execute("SELECT vector FROM metadata")
        rows = cur.fetchall()
        if not rows:
            return

        vectors = np.vstack(
            [np.frombuffer(r[0], dtype="float32") for r in rows]
        )

        if self.index is None:
            self._create_index(vectors.shape[1])

        self._train_and_add(vectors)

    def _train_and_add(self, vectors: np.ndarray):
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)
        self._train_buffer.clear()

    # ------------------------------------------------------------------
    # Add vectors (WRITE MODE ONLY)
    # ------------------------------------------------------------------
    def add(self, vectors: np.ndarray, metas: List[Dict]) -> int:
        if self.readonly:
            raise RuntimeError("FaissStore is readonly; cannot add vectors.")

        if self.max_vectors is not None:
            remaining = self.max_vectors - self.count_vectors()
            if remaining <= 0:
                return 0
            vectors = vectors[:remaining]
            metas = metas[:remaining]

        vectors = vectors.astype("float32")

        if self.index is None:
            self._create_index(vectors.shape[1])

        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self._train_buffer.append(vectors)
            self._train_and_add(np.vstack(self._train_buffer))
        else:
            self.index.add(vectors)

        cur = self.conn.cursor()
        for vec, meta in zip(vectors, metas):
            cur.execute(
                """
                INSERT INTO metadata (chunk_id, original_id, text, source, vector)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    meta["id"],
                    meta.get("original_id"),
                    meta["text"],
                    meta.get("source"),
                    vec.tobytes(),
                ),
            )

        self.conn.commit()
        return len(vectors)

    # ------------------------------------------------------------------
    # Search (READ MODE)
    # ------------------------------------------------------------------
    def search(self, query: np.ndarray, k: int) -> Tuple[List[float], List[Dict]]:
        if not self.is_ready():
            return [], []

        query = query.astype("float32")
        distances, ids = self.index.search(query, k)

        cur = self.conn.cursor()
        results = []

        for idx in ids[0]:
            if idx < 0:
                continue

            cur.execute(
                """
                SELECT chunk_id, original_id, text, source
                FROM metadata
                LIMIT 1 OFFSET ?
                """,
                (int(idx),),
            )
            row = cur.fetchone()
            if row:
                results.append(
                    {
                        "id": row[0],
                        "original_id": row[1],
                        "text": row[2],
                        "source": row[3],
                    }
                )

        return distances[0].tolist(), results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_vectors(self) -> int:
        if self.index:
            return int(self.index.ntotal)
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM metadata")
        return int(cur.fetchone()[0])

    def is_ready(self) -> bool:
        return (
            self.index is not None
            and self.count_vectors() > 0
            and (not hasattr(self.index, "is_trained") or self.index.is_trained)
        )

    @property
    def is_full(self) -> bool:
        return (
            self.max_vectors is not None
            and self.count_vectors() >= self.max_vectors
        )
    
    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------
    def _open_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)

    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                original_id TEXT,
                text TEXT NOT NULL,
                source TEXT,
                vector BLOB NOT NULL
            )
            """
        )
        self.conn.commit()


