"""Hybrid retriever combining BM25 and dense retrieval with fixed convex combination."""

from typing import Dict, List, Optional
from router_retrieval.data.schema import Candidate, Query
from router_retrieval.retrieval.bm25 import BM25Retriever
from router_retrieval.retrieval.dense import DenseRetriever
from router_retrieval.retrieval.normalization import normalize_scores


# Raw BM25 scores are >= 0 and BM25Retriever only returns positively-scored documents, so a BM25
# list shorter than k means every unretrieved document has a raw score of exactly 0.
BM25_TRUNCATED_FLOOR = 0.0


def normalize_for_fusion(
    candidates: List[Candidate],
    k: int,
    method: str = "min_max",
    floor_if_truncated: Optional[float] = None,
) -> Dict[str, float]:
    """
    Normalize one retriever's list into a {doc_id: score} map for fusion.

    If the list holds fewer than k candidates and the score of every unretrieved document is
    known (`floor_if_truncated`), min-max is anchored at that floor instead of the list minimum,
    so retrieved documents always outrank unretrieved ones for that retriever.
    """
    floor = floor_if_truncated if len(candidates) < k else None
    return {c.doc_id: c.score for c in normalize_scores(candidates, method, floor)}


def fuse_normalized_scores(
    query_id: str,
    bm25_norm: Dict[str, float],
    dense_norm: Dict[str, float],
    alpha: float,
    k: int = 100,
    missing_score: float = 0.0,
) -> List[Candidate]:
    """
    Convex-combine two already-normalized {doc_id: score} maps over their union.

    A document absent from one retriever's candidate list receives `missing_score` for that
    retriever (0.0 == the normalization floor, which the lowest retrieved document can share).
    Exact ties in the fused score are broken by (1) the alpha-weighted count of retrievers that
    actually returned the document, (2) the unweighted sum of normalized scores, (3) doc_id -- so
    alpha in {0, 1} reproduces the corresponding single-retriever ranking exactly.
    """
    fused = {}
    for d in bm25_norm.keys() | dense_norm.keys():
        s_dense, s_bm25 = dense_norm.get(d, missing_score), bm25_norm.get(d, missing_score)
        retrieved_weight = alpha * (d in dense_norm) + (1.0 - alpha) * (d in bm25_norm)
        fused[d] = (alpha * s_dense + (1.0 - alpha) * s_bm25, retrieved_weight, s_dense + s_bm25)
    ranked = sorted(fused.items(), key=lambda p: (-p[1][0], -p[1][1], -p[1][2], p[0]))[:k]
    return [Candidate(query_id=query_id, doc_id=d, score=s[0], rank=r + 1) for r, (d, s) in enumerate(ranked)]


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
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
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
        return self.combine_candidates(self.bm25.search(query, k), self.dense.search(query, k), k)

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
        source = bm25_candidates or dense_candidates
        if not source:
            return []
        bm25_norm = normalize_for_fusion(bm25_candidates, k, self.normalization_method, BM25_TRUNCATED_FLOOR)
        dense_norm = normalize_for_fusion(dense_candidates, k, self.normalization_method)
        return fuse_normalized_scores(source[0].query_id, bm25_norm, dense_norm, self.alpha, k)
