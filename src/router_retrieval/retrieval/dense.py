"""Dense embedding retrieval using SentenceTransformers and FAISS."""

from pathlib import Path
from typing import List, Dict, Optional, Sequence
from router_retrieval.data.schema import Document, Candidate, Query


class DenseRetriever:
    """Dense retriever using bi-encoder representations and FAISS CPU index."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/msmarco-distilbert-base-v4",
        batch_size: int = 64,
        device: str = "cpu",
        faiss_index_type: str = "IndexFlatIP",
    ):
        """Initialize dense retriever configuration."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.faiss_index_type = faiss_index_type

    def index(self, corpus: Dict[str, Document]) -> None:
        """
        Encode corpus passages and build FAISS index.

        Args:
            corpus: Dict of doc_id to Document.
        """
        raise NotImplementedError

    def search(self, query: Query, k: int = 100) -> List[Candidate]:
        """
        Encode query and retrieve top-k nearest neighbor candidates from FAISS index.

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
        """Persist FAISS index and doc_id mapping to disk."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path, model_name: Optional[str] = None) -> "DenseRetriever":
        """Load FAISS index and doc_id mapping from disk."""
        raise NotImplementedError
