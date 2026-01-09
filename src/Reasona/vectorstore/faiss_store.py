from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss
import sqlite3
import json


class FaissStore:
    """
    Disk-backed FAISS store with SQLite metadata storage.
    Safe for multi-million scale IVF indexing.
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
        max_vectors: Optional[int] = None,
    ):
        self.dim = dim
        self.index_path = index_path
        self.db_path = db_path
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq = pq
        self.train_threshold = max(train_threshold, nlist)
        self.max_vectors = max_vectors

        # ---------- FAISS ----------
        quantizer = faiss.IndexFlatL2(dim)
        if pq:
            self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq, 8)
        else:
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

        self.index.nprobe = nprobe
        self.is_trained = False

        # ---------- Buffers ----------
        self._train_vectors: List[np.ndarray] = []
        self._pending_vectors: List[np.ndarray] = []

        # ---------- SQLite ----------
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    # -------------------------
    # Properties
    # -------------------------
    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    @property
    def is_full(self) -> bool:
        return self.max_vectors is not None and self.ntotal >= self.max_vectors

    # -------------------------
    # Add vectors
    # -------------------------
    def add(self, vectors: np.ndarray, metas: List[Dict]) -> int:
        if self.is_full:
            return 0

        vectors = vectors.astype("float32")

        if self.max_vectors is not None:
            remaining = self.max_vectors - self.ntotal
            if remaining <= 0:
                return 0
            vectors = vectors[:remaining]
            metas = metas[:remaining]

        # 1. Insert metadata FIRST (IDs must match vector order)
        metas_json = [(json.dumps(m),) for m in metas]
        self.cursor.executemany(
            "INSERT INTO metadata (data) VALUES (?)",
            metas_json,
        )
        self.conn.commit()

        # 2. Handle training vs adding
        if not self.is_trained:
            self._train_vectors.append(vectors)
            self._pending_vectors.append(vectors)
            self._train_once()
            return len(vectors)

        self.index.add(vectors)
        return len(vectors)

    # -------------------------
    # Training
    # -------------------------
    def _train_once(self):
        if self.is_trained:
            return

        total = sum(v.shape[0] for v in self._train_vectors)
        if total < self.train_threshold:
            return

        train_vectors = np.vstack(self._train_vectors)
        self.index.train(train_vectors)

        # IMPORTANT: add all buffered vectors AFTER training
        pending = np.vstack(self._pending_vectors)
        self.index.add(pending)

        self._train_vectors.clear()
        self._pending_vectors.clear()
        self.is_trained = True

    # -------------------------
    # Search
    # -------------------------
    def search(self, query: np.ndarray, k: int = 5):
        if not self.is_trained or self.ntotal == 0:
            return [], []

        query = query.astype("float32")
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

    # -------------------------
    # Persistence
    # -------------------------
    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.conn.commit()

    def load(self):
        if self.index_path.exists():

            self.index = faiss.read_index(str(self.index_path))
            self.index.nprobe = self.nprobe
            self.index.is_trained = self.is_trained

    def _finalize(self):
        """
        Call once at the very end.
        """
        if not self.is_trained and self._pending_vectors:
            self._train_once()

        self.save()

    def close(self):
        self.conn.close()
