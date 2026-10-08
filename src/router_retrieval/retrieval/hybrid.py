"""Hybrid retriever combining BM25 and dense retrieval with fixed convex combination."""

from typing import List
from router_retrieval.data.schema import Candidate, Query
from router_retrieval.retrieval.bm25 import BM25Retriever
from router_retrieval.retrieval.dense import DenseRetriever


class HybridRetriever:
    """
    Fixed-alpha convex combination hybrid retriever.
    score = alpha * norm_dense + (1 - alpha) * norm_bm25
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        alpha: float = 0.5,
        normalization_method: str = "min_max",
    ):
        """
        Initialize HybridRetriever.

        Args:
            bm25_retriever: Initialized or indexed BM25 retriever.
            dense_retriever: Initialized or indexed dense retriever.
            alpha: Convex weight on dense retriever (1 - alpha on BM25).
            normalization_method: Method for score normalization before combination.
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.normalization_method = normalization_method

    def search(self, query: Query, k: int = 100) -> List[Candidate]:
        """
        Execute BM25 and dense retrieval, normalize, combine, and rank top-k.

        Args:
            query: Query dataclass.
            k: Number of candidates to return.

        Returns:
            List of combined Candidates sorted by hybrid score descending.
        """
        raise NotImplementedError

    def combine_candidates(
        self,
        bm25_candidates: List[Candidate],
        dense_candidates: List[Candidate],
        k: int = 100,
    ) -> List[Candidate]:
        """
        Combine precomputed BM25 and dense candidate lists.

        Args:
            bm25_candidates: BM25 candidate list.
            dense_candidates: Dense candidate list.
            k: Top-k candidates to retain.

        Returns:
            List of fused Candidates.
        """
        raise NotImplementedError
