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
        if k < 0:
            raise ValueError("k must be non-negative")
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
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not candidate_rankings or top_k == 0:
            return []

        # Find query_id from the first non-empty candidate list
        query_id = None
        for ranking in candidate_rankings:
            if ranking:
                query_id = ranking[0].query_id
                break

        if query_id is None:
            return []

        # Accumulate RRF score and appearance count per doc_id
        doc_scores = {}
        for ranking in candidate_rankings:
            for rank_idx, cand in enumerate(ranking):
                if cand.query_id != query_id:
                    raise ValueError("All candidates must belong to the same query")
                r = cand.rank if (cand.rank and cand.rank > 0) else (rank_idx + 1)
                score, count = doc_scores.get(cand.doc_id, (0.0, 0))
                doc_scores[cand.doc_id] = (score + 1.0 / (self.k + r), count + 1)

        # Sort: score descending, appearance count descending, doc_id ascending
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda p: (-p[1][0], -p[1][1], p[0]),
        )[:top_k]

        return [
            Candidate(query_id=query_id, doc_id=doc_id, score=float(scores[0]), rank=r + 1)
            for r, (doc_id, scores) in enumerate(sorted_docs)
        ]
