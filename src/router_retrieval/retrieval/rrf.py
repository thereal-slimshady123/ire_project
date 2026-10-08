"""Reciprocal Rank Fusion (RRF) rank-based baseline."""

from typing import List, Sequence
from router_retrieval.data.schema import Candidate


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) algorithm:
    RRF_score(d) = sum_{m in models} 1 / (k + rank_m(d))
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF.

        Args:
            k: Ranking constant (default 60).
        """
        self.k = k

    def fuse(
        self,
        candidate_rankings: Sequence[List[Candidate]],
        top_k: int = 100,
    ) -> List[Candidate]:
        """
        Fuse multiple candidate rankings for the same query.

        Args:
            candidate_rankings: Sequence of candidate rankings (e.g., [bm25_list, dense_list]).
            top_k: Number of fused results to return.

        Returns:
            List of fused Candidate objects sorted by RRF score descending.
        """
        raise NotImplementedError
