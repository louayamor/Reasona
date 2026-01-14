from pathlib import Path
from typing import Dict, List, Tuple
import sqlite3
import threading
import numpy as np
import faiss
import os
from typing import Optional

faiss.omp_set_num_threads(1)


class ShardedFaissStore:
    """
    Independent sharded FAISS store.
    Each shard has:
    - Its own FAISS index file
    - Its own SQLite metadata DB
    """

    def __init__(
        self,
        dim: int,
        base_path: Path,
        max_vectors_per_shard: int = 100_000,
        nprobe: int = 16,
        mmap: bool = True,
    ):
        self.dim = dim
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_vectors_per_shard = max_vectors_per_shard
        self.nprobe = nprobe
        self.mmap = mmap

        self.shards: List[Dict] = []  # Each shard: {"index": faiss.Index, "db_path": Path, "index_path": Path}
        self._conn_thread: Dict[int, sqlite3.Connection] = {}  # For multithreading
        self._load_existing_shards()

    # ------------------------
    # Shard management
    # ------------------------
    def _load_existing_shards(self):
        shard_files = sorted(self.base_path.glob("shard_*.faiss"))
        for idx, shard_file in enumerate(shard_files):
            db_file = self.base_path / f"shard_{idx}.db"
            index = self._load_index(shard_file)
            self.shards.append({"index": index, "index_path": shard_file, "db_path": db_file})
        print(f"Loaded {len(self.shards)} existing shards.")

    def _get_active_shard(self):
        if not self.shards or self._is_full(self.shards[-1]):
            shard_id = len(self.shards)
            index_path = self.base_path / f"shard_{shard_id}.faiss"
            db_path = self.base_path / f"shard_{shard_id}.db"
            index = self._create_index()
            self._init_db(db_path)
            self.shards.append({"index": index, "index_path": index_path, "db_path": db_path})
        return self.shards[-1]

    def _is_full(self, shard: Dict):
        conn = self._get_conn(shard["db_path"])
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM metadata")
        count = cur.fetchone()[0]
        return count >= self.max_vectors_per_shard

    # ------------------------
    # FAISS helpers
    # ------------------------
    def _create_index(self):
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, 1024, faiss.METRIC_L2)
        index.nprobe = self.nprobe
        return index

    def _load_index(self, path: Path):
        if path.exists():
            return faiss.read_index(str(path))
        return self._create_index()

    # ------------------------
    # DB helpers
    # ------------------------
    def _init_db(self, db_path: Path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
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
        conn.commit()
        conn.close()

    def _get_conn(self, db_path: Path):
        thread_id = threading.get_ident()
        key = (thread_id, str(db_path))
        if key not in self._conn_thread:
            self._conn_thread[key] = sqlite3.connect(db_path)
        return self._conn_thread[key]

    # ------------------------
    # Add vectors
    # ------------------------
    def add(self, vectors: np.ndarray, metas: List[Dict]) -> int:
        vectors = vectors.astype("float32")
        start = 0
        total_added = 0
        while start < len(vectors):
            shard = self._get_active_shard()
            remaining = self.max_vectors_per_shard - self.count_vectors(shard)
            end = min(start + remaining, len(vectors))

            # Train index if needed
            index = shard["index"]
            if hasattr(index, "is_trained") and not index.is_trained:
                index.train(vectors[start:end])
            index.add(vectors[start:end])

            # Store metadata
            conn = self._get_conn(shard["db_path"])
            cur = conn.cursor()
            for vec, meta in zip(vectors[start:end], metas[start:end]):
                cur.execute(
                    """
                    INSERT INTO metadata (chunk_id, original_id, text, source, vector)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (meta["id"], meta.get("original_id"), meta["text"], meta.get("source"), vec.tobytes()),
                )
            conn.commit()

            total_added += (end - start)
            start = end
        return total_added

    # ------------------------
    # Search
    # ------------------------
    def search(self, query: np.ndarray, k: int) -> Tuple[List[float], List[Dict]]:
        distances_list = []
        results_list = []

        query = query.astype("float32")
        for shard in self.shards:
            index = shard["index"]
            d, ids = index.search(query, k)
            conn = self._get_conn(shard["db_path"])
            cur = conn.cursor()
            for i, idx in enumerate(ids[0]):
                if idx < 0:
                    continue
                cur.execute("SELECT chunk_id, original_id, text, source FROM metadata LIMIT 1 OFFSET ?", (idx,))
                row = cur.fetchone()
                if row:
                    results_list.append({"id": row[0], "original_id": row[1], "text": row[2], "source": row[3]})
                    distances_list.append(d[0][i])

        # Top-k across all shards
        if len(distances_list) <= k:
            return distances_list, results_list
        idxs = np.argsort(distances_list)[:k]
        return [distances_list[i] for i in idxs], [results_list[i] for i in idxs]

    # ------------------------
    # Utilities
    # ------------------------
    def save(self):
        for shard in self.shards:
            faiss.write_index(shard["index"], str(shard["index_path"]))

    def finalize(self):
        self.save()
        for key, conn in self._conn_thread.items():
            conn.close()
        self._conn_thread.clear()

    def count_vectors(self, shard: Optional[Dict] = None) -> int:
        if shard:
            conn = self._get_conn(shard["db_path"])
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM metadata")
            return cur.fetchone()[0]
        return sum(self.count_vectors(sh) for sh in self.shards)

    @property
    def is_full(self) -> bool:
        return all(self.count_vectors(sh) >= self.max_vectors_per_shard for sh in self.shards)
