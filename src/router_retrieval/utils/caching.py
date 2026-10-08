"""Disk caching utilities for embeddings, features, and retrieval results."""

from pathlib import Path
from typing import Any, Optional


class DiskCache:
    """Persistent on-disk cache for intermediate artifacts using pickle/parquet."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached object by key, or return None if absent."""
        raise NotImplementedError

    def set(self, key: str, value: Any) -> None:
        """Store object in disk cache with specified key."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        """Check if cached key exists."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear cache directory."""
        raise NotImplementedError
