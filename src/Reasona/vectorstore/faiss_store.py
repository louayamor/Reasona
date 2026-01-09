from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss
import sqlite3
import json


class FaissStore:
    """
    Disk-backed FAISS store with SQLite metadata storage.
    Designed for multi-million scale.
    """

    def __init__(
        self,
        dim: int,
        index_path: Path,
        db_path: Path,
        nlist: int = 1024,
        nprobe: int = 16,
        pq: Optional[int] = None,
        train_threshold: int = 50_000,
        max_vectors: Optional[int] = None,  # NEW
    ):
        self.dim = dim
        self.index_path = index_path
        self.db_path = db_path
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq = pq
        self.train_threshold = max(train_threshold, nlist)
        self.max_vectors = max_vectors

        self._train_buffer: List[np.ndarray] = []

        self.quantizer = faiss.IndexFlatL2(dim)

        if pq:
            self.index = faiss.IndexIVFPQ(
                self.quantizer, dim, nlist, pq, 8
            )
        else:
            self.index = faiss.IndexIVFFlat(
                self.quantizer, dim, nlist, faiss.METRIC_L2
            )

        self.index.nprobe = self.nprobe
        self.is_trained = False

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT
            )
        """)
        self.conn.commit()

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    @property
    def is_full(self) -> bool:
        return self.max_vectors is not None and self.ntotal >= self.max_vectors

    def _train_once(self):
        if self.is_trained:
            return
        total_vectors = sum(v.shape[0] for v in self._train_buffer)
        if total_vectors < self.train_threshold:
            return

        train_vectors = np.vstack(self._train_buffer).astype("float32")
        self.index.train(train_vectors)
        self.is_trained = True
        self._train_buffer.clear()

    def add(self, vectors: np.ndarray, metas: List[Dict]):
        if self.is_full:
            return 0  

        vectors = vectors.astype("float32")

        if self.max_vectors is not None:
            remaining = self.max_vectors - self.ntotal
            if remaining <= 0:
                return 0
            vectors = vectors[:remaining]
            metas = metas[:remaining]

        metas_json = [(json.dumps(m),) for m in metas]
        self.cursor.executemany(
            "INSERT INTO metadata (data) VALUES (?)", metas_json
        )
        self.conn.commit()

        if not self.is_trained:
            self._train_buffer.append(vectors)
            self._train_once()
            return len(vectors)

        self.index.add(vectors)
        return len(vectors)

    def search(self, query: np.ndarray, k: int = 5):
        query = query.astype("float32")
        if not self.is_trained or self.ntotal == 0:
            return [], []

        distances, indices = self.index.search(query, k)
        results = []

        for idx in indices[0]:
            if idx == -1:
                continue
            self.cursor.execute(
                "SELECT data FROM metadata WHERE id=?",
                (idx + 1,),
            )
            row = self.cursor.fetchone()
            if row:
                results.append(json.loads(row[0]))

        return distances[0], results

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.conn.commit()

    def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(
                str(self.index_path), faiss.IO_FLAG_MMAP
            )
            self.is_trained = self.index.is_trained

    def _finalize(self):
        if not self.is_trained and self._train_buffer:
            train_vectors = np.vstack(self._train_buffer).astype("float32")
            self.index.train(train_vectors)
            self.is_trained = True
            self._train_buffer.clear()

    def close(self):
        self.conn.close()
