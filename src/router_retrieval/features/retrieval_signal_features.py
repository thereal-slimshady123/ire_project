"""Retrieval signal feature extraction comparing BM25 and dense candidate outputs."""

from typing import List, Dict
from router_retrieval.data.schema import Candidate


def compute_jaccard_overlap(
    bm25_candidates: List[Candidate],
    dense_candidates: List[Candidate],
    k: int = 10,
) -> float:
    """Compute set Jaccard similarity between top-k BM25 and dense retrieved doc IDs."""
    raise NotImplementedError


def compute_rank_correlation(
    bm25_candidates: List[Candidate],
    dense_candidates: List[Candidate],
    k: int = 20,
    method: str = "spearman",
) -> float:
    """Compute rank correlation (Spearman or Kendall tau) over candidate intersection."""
    raise NotImplementedError


def compute_score_entropy(candidates: List[Candidate]) -> float:
    """Compute Shannon entropy over normalized candidate score distribution."""
    raise NotImplementedError


def compute_top_gap(candidates: List[Candidate]) -> float:
    """Compute margin between top-1 and top-2 candidate scores."""
    raise NotImplementedError


def extract_retrieval_signal_features(
    bm25_candidates: List[Candidate],
    dense_candidates: List[Candidate],
) -> Dict[str, float]:
    """
    Extract retrieval-signal features from sparse and dense candidate lists.

    Features extracted:
        - jaccard_overlap_top10, jaccard_overlap_top20
        - rank_correlation_spearman
        - bm25_top1_score, bm25_top1_top2_gap, bm25_score_entropy
        - dense_top1_score, dense_top1_top2_gap, dense_score_entropy

    Args:
        bm25_candidates: Ranked list from BM25.
        dense_candidates: Ranked list from Dense retriever.

    Returns:
        Dictionary of feature names to numeric values.
    """
    raise NotImplementedError
