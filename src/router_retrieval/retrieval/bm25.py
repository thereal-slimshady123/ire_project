"""BM25 sparse retrieval implementation."""

from pathlib import Path
from typing import List, Dict, Optional, Sequence
from router_retrieval.data.schema import Document, Candidate, Query


class BM25Retriever:
    """BM25 retrieval engine."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 retriever with tuning parameters."""
        self.k1 = k1
        self.b = b
        self.corpus: Optional[Dict[str, Document]] = None

    def index(self, corpus: Dict[str, Document]) -> None:
        """
        Build BM25 index from document collection.

        Args:
            corpus: Dict of doc_id to Document.
        """
        raise NotImplementedError

    def search(self, query: Query, k: int = 100) -> List[Candidate]:
        """
        Retrieve top-k candidates for a given query.

        Args:
            query: Query dataclass.
            k: Number of candidates to retrieve.

        Returns:
            List of Candidate objects sorted by score descending.
        """
        raise NotImplementedError

    def batch_search(self, queries: Sequence[Query], k: int = 100) -> Dict[str, List[Candidate]]:
        """Batch search for multiple queries."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Persist BM25 index to disk."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        """Load BM25 index from disk."""
        raise NotImplementedError
