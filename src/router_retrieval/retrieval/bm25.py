"""BM25 sparse retrieval implementation (bm25s backend)."""

from pathlib import Path
from typing import List, Dict, Optional, Sequence
from router_retrieval.data.schema import Document, Candidate, Query


class BM25Retriever:
    """BM25 retrieval engine backed by bm25s, with English stemming and stopword removal."""

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        stemmer: Optional[str] = "english",
        stopwords: Optional[str] = "en",
    ):
        """Initialize BM25 retriever with tuning parameters and tokenization settings."""
        self.k1 = k1
        self.b = b
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.corpus: Optional[Dict[str, Document]] = None

    def index(self, corpus: Dict[str, Document]) -> None:
        """
        Build BM25 index from document collection (indexes Document.full_text).

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
            List of Candidate objects sorted by score descending (rank starts at 1).
            An empty or fully out-of-vocabulary query returns an empty list.
        """
        raise NotImplementedError

    def batch_search(self, queries: Sequence[Query], k: int = 100) -> Dict[str, List[Candidate]]:
        """Batch search for multiple queries."""
        raise NotImplementedError

    def score_docs(self, query: Query, doc_ids: Sequence[str]) -> Dict[str, float]:
        """
        Exact raw BM25 score of the query against specific documents (0.0 if no term overlap).
        Used by the hybrid retriever to score documents that only the dense retriever returned.
        """
        raise NotImplementedError

    def idf_vocab(self) -> Dict[str, float]:
        """Per-term IDF from the indexed corpus (after stemming), used by query features."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Persist BM25 index to disk."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        """Load BM25 index from disk."""
        raise NotImplementedError
