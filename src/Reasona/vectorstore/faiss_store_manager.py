import faiss
from pathlib import Path
import shutil
from datetime import datetime
from typing import Optional

class FaissStoreManager:

    def __init__(self, store_dir: str, base_name: str = "faiss_index", keep_versions: int = 5):
        self.store_dir = Path(store_dir)
        self.base_name = base_name
        self.keep_versions = keep_versions
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_index_path(self, version: Optional[str] = None) -> Path:
        if version is None:
            version = self._get_timestamp()
        return self.store_dir / f"{self.base_name}_{version}.index"

    def save_index(self, index: faiss.Index, version: Optional[str] = None) -> Path:
        path = self._get_index_path(version)
        faiss.write_index(index, str(path))
        self._cleanup_old_versions()
        return path

    def load_latest_index(self) -> Optional[faiss.Index]:
        indices = sorted(self.store_dir.glob(f"{self.base_name}_*.index"), reverse=True)
        if not indices:
            return None
        return faiss.read_index(str(indices[0]))

    def _cleanup_old_versions(self):
        indices = sorted(self.store_dir.glob(f"{self.base_name}_*.index"), reverse=True)
        for old_index in indices[self.keep_versions:]:
            old_index.unlink()
