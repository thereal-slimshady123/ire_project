"""Dense embedding retrieval using SentenceTransformers and FAISS."""

from pathlib import Path
from typing import List, Dict, Optional, Sequence
from router_retrieval.data.schema import Document, Candidate, Query


class DenseRetriever:
    """
    Dense retriever using bi-encoder representations and an exact FAISS inner-product index.
    The default model (TAS-B) is trained for dot-product similarity, so embeddings are not normalized.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/msmarco-distilbert-base-tas-b",
        batch_size: int = 64,
        device: str = "cuda",
        faiss_index_type: str = "IndexFlatIP",
        normalize_embeddings: bool = False,
    ):
        """Initialize dense retriever configuration (device falls back to cpu without CUDA)."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.faiss_index_type = faiss_index_type
        self.normalize_embeddings = normalize_embeddings

    def index(self, corpus: Dict[str, Document]) -> None:
        """
        Encode corpus passages (Document.full_text) and build FAISS index.

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
            List of Candidate objects sorted by score descending (rank starts at 1).
        """
        raise NotImplementedError

    def batch_search(self, queries: Sequence[Query], k: int = 100) -> Dict[str, List[Candidate]]:
        """Batch search for multiple queries."""
        raise NotImplementedError

    def score_docs(self, query: Query, doc_ids: Sequence[str]) -> Dict[str, float]:
        """
        Exact dot-product score of the query embedding against stored document embeddings.
        Used by the hybrid retriever to score documents that only BM25 returned.
        """
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Persist FAISS index, document embeddings and doc_id mapping to disk."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path, model_name: Optional[str] = None) -> "DenseRetriever":
        """Load FAISS index, document embeddings and doc_id mapping from disk."""
        raise NotImplementedError
