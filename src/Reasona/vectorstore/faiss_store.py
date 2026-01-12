from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sqlite3
import json
import numpy as np
import faiss

from Reasona.utils.logger import setup_logger


class FaissStore:

    def __init__(
        self,
        dim: Optional[int],
        index_path: Path,
        db_path: Path,
        max_vectors: Optional[int] = None,
        nprobe: int = 16,
        mmap: bool = True,
    ):
        self.dim = dim
        self.index_path = index_path
        self.db_path = db_path
        self.max_vectors = max_vectors
        self.nprobe = nprobe
        self.mmap = mmap

        self.index: Optional[faiss.Index] = None
        self.conn: Optional[sqlite3.Connection] = None

        self.logger = setup_logger(
            "faiss_store",
            "logs/vectorstore/faiss_store.json",
        )

        self.logger.info(
            "FaissStore init | index=%s | db=%s | max_vectors=%s | mmap=%s",
            self.index_path,
            self.db_path,
            self.max_vectors,
            self.mmap,
        )

    def load(self):
        self._open_db()
        self._ensure_schema()

        if self.index_path.exists():
            flags = faiss.IO_FLAG_MMAP if self.mmap else 0
            self.logger.info("Loading FAISS index | mmap=%s", self.mmap)

            self.index = faiss.read_index(str(self.index_path), flags)

            if hasattr(self.index, "nprobe"):
                self.index.nprobe = self.nprobe

            self.logger.info(
                "FAISS loaded | ntotal=%d | trained=%s",
                self.index.ntotal,
                getattr(self.index, "is_trained", True),
            )

            if hasattr(self.index, "is_trained") and not self.index.is_trained:
                self.logger.warning("Index untrained after load | retraining")
                self._retrain_from_db()
        else:
            self.logger.info("No FAISS index found | will create on first add")

    def close(self):
        if self.conn:
            self.conn.close()
            self.logger.info("SQLite connection closed")

    def save(self):
        if self.index:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            self.logger.info(
                "FAISS index saved | ntotal=%d",
                self.index.ntotal,
            )

    def finalize(self):
        self.save()
        self.close()

    def _create_index(self, dim: int):
        self.dim = dim

        if self.max_vectors:
            nlist = min(4096, max(128, int(np.sqrt(self.max_vectors))))
        else:
            nlist = 1024

        self.logger.info(
            "Creating IVF index | dim=%d | nlist=%d",
            dim,
            nlist,
        )

        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(
            quantizer,
            dim,
            nlist,
            faiss.METRIC_L2,
        )
        self.index.nprobe = self.nprobe

    def _retrain_from_db(self):
        cur = self.conn.cursor()
        cur.execute("SELECT vector FROM metadata")
        rows = cur.fetchall()

        if not rows:
            self.logger.warning("No vectors in DB | retrain skipped")
            return

        vectors = np.vstack(
            [np.frombuffer(r[0], dtype="float32") for r in rows]
        )

        self.logger.info(
            "Retraining FAISS | vectors=%d | dim=%d",
            vectors.shape[0],
            vectors.shape[1],
        )

        if self.index is None:
            self._create_index(vectors.shape[1])

        self.index.train(vectors)
        self.index.add(vectors)

        self.logger.info(
            "Retrain complete | ntotal=%d",
            self.index.ntotal,
        )

    def add(self, vectors: np.ndarray, metas: List[Dict]) -> int:
        if self.max_vectors is not None:
            remaining = self.max_vectors - self.count_vectors()
            if remaining <= 0:
                self.logger.warning("Store full | add skipped")
                return 0

            vectors = vectors[:remaining]
            metas = metas[:remaining]

        vectors = vectors.astype("float32")

        if self.index is None:
            self._create_index(vectors.shape[1])

        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.logger.info("Training FAISS on first batch")
            self.index.train(vectors)

        self.index.add(vectors)

        cur = self.conn.cursor()
        for vec, meta in zip(vectors, metas):
            cur.execute(
                "INSERT INTO metadata (vector, data) VALUES (?, ?)",
                (vec.tobytes(), json.dumps(meta)),
            )
        self.conn.commit()

        self.logger.info(
            "Vectors added | added=%d | ntotal=%d",
            len(vectors),
            self.index.ntotal,
        )

        return len(vectors)

    def search(self, query: np.ndarray, k: int) -> Tuple[List[float], List[Dict]]:
        if not self.is_ready():
            self.logger.error("Search attempted on unready index")
            return [], []

        query = query.astype("float32")
        distances, ids = self.index.search(query, k)

        cur = self.conn.cursor()
        results = []

        for idx in ids[0]:
            if idx < 0:
                continue
            cur.execute(
                "SELECT data FROM metadata LIMIT 1 OFFSET ?",
                (int(idx),),
            )
            row = cur.fetchone()
            if row:
                results.append(json.loads(row[0]))

        self.logger.info(
            "Search executed | returned=%d",
            len(results),
        )

        return distances[0].tolist(), results
    
    def count_vectors(self) -> int:
        if self.index:
            return int(self.index.ntotal)

        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM metadata")
        return int(cur.fetchone()[0])

    @property
    def is_full(self) -> bool:
        return self.max_vectors is not None and self.count_vectors() >= self.max_vectors

    def is_ready(self) -> bool:
        return (
            self.index is not None
            and self.count_vectors() > 0
            and (not hasattr(self.index, "is_trained") or self.index.is_trained)
        )

    def _open_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.logger.info("SQLite opened | path=%s", self.db_path)

    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector BLOB NOT NULL,
                data TEXT NOT NULL
            )
            """
        )
        self.conn.commit()
