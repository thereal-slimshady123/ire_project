"""Retrieval signal feature extraction comparing BM25 and dense candidate outputs."""

import math
from typing import Iterable, List, Dict, Set

import numpy as np
from scipy import stats

from router_retrieval.data.schema import Candidate
from router_retrieval.retrieval.normalization import min_max_normalize


def _top_ids(candidates: List[Candidate], k: int) -> List[str]:
    return [c.doc_id for c in sorted(candidates, key=lambda c: c.rank)[:k]]


def compute_jaccard_overlap(
    bm25_candidates: List[Candidate],
    dense_candidates: List[Candidate],
    k: int = 10,
) -> float:
    """Compute set Jaccard similarity between top-k BM25 and dense retrieved doc IDs."""
    a, b = set(_top_ids(bm25_candidates, k)), set(_top_ids(dense_candidates, k))
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def compute_rank_correlation(
    bm25_candidates: List[Candidate],
    dense_candidates: List[Candidate],
    k: int = 20,
    method: str = "spearman",
) -> float:
    """
    Compute rank correlation (Spearman or Kendall tau) over candidate intersection.

    Returns 0.0 when fewer than 3 documents are shared (correlation undefined / meaningless).
    """
    rank_a = {d: i for i, d in enumerate(_top_ids(bm25_candidates, k))}
    rank_b = {d: i for i, d in enumerate(_top_ids(dense_candidates, k))}
    shared = sorted(rank_a.keys() & rank_b.keys())
    if len(shared) < 3:
        return 0.0
    x = [rank_a[d] for d in shared]
    y = [rank_b[d] for d in shared]
    if method == "spearman":
        r = stats.spearmanr(x, y).statistic
    elif method == "kendall":
        r = stats.kendalltau(x, y).statistic
    else:
        raise ValueError(f"Unknown rank correlation method: {method}")
    return 0.0 if np.isnan(r) else float(r)


def compute_score_entropy(candidates: List[Candidate], k: int = 10) -> float:
    """
    Compute Shannon entropy over normalized candidate score distribution.

    Scores of the top-k candidates are min-max normalized over the full list, turned into a
    distribution, and the entropy is divided by log(k) so the result lies in [0, 1]
    (1 == flat, uninformative ranking).
    """
    if len(candidates) < 2:
        return 0.0
    norm = min_max_normalize(candidates)
    top = np.array([c.score for c in sorted(norm, key=lambda c: c.rank)[:k]], dtype=np.float64)
    if top.sum() <= 0:
        return 1.0
    p = top / top.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(len(top)))


def compute_top_gap(candidates: List[Candidate]) -> float:
    """Compute margin between top-1 and top-2 candidate scores."""
    if len(candidates) < 2:
        return 0.0
    ranked = sorted(candidates, key=lambda c: c.rank)
    return float(ranked[0].score - ranked[1].score)


def compute_top_ratio(candidates: List[Candidate]) -> float:
    """Ratio of top-1 to top-2 raw score (1.0 when undefined)."""
    if len(candidates) < 2:
        return 1.0
    ranked = sorted(candidates, key=lambda c: c.rank)
    return float(ranked[0].score / ranked[1].score) if ranked[1].score > 0 else 1.0


def compute_query_term_coverage(query_terms: Iterable[str], doc_term_sets: List[Set[str]]) -> float:
    """Mean fraction of unique (analyzed) query terms that appear in each document."""
    terms = set(query_terms)
    if not terms or not doc_term_sets:
        return 0.0
    return float(np.mean([len(terms & d) / len(terms) for d in doc_term_sets]))


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

    Top-1 scores are raw; top1-top2 gaps are on per-query min-max normalized scores, i.e. the
    scale on which the hybrid fusion actually sees them, so gaps are comparable across retrievers.

    Args:
        bm25_candidates: Ranked list from BM25.
        dense_candidates: Ranked list from Dense retriever.

    Returns:
        Dictionary of feature names to numeric values.
    """
    feats = {
        "jaccard_overlap_top10": compute_jaccard_overlap(bm25_candidates, dense_candidates, 10),
        "jaccard_overlap_top20": compute_jaccard_overlap(bm25_candidates, dense_candidates, 20),
        "rank_correlation_spearman": compute_rank_correlation(bm25_candidates, dense_candidates, 20),
    }
    for name, cands in (("bm25", bm25_candidates), ("dense", dense_candidates)):
        ranked = sorted(cands, key=lambda c: c.rank)
        feats[f"{name}_top1_score"] = float(ranked[0].score) if ranked else 0.0
        feats[f"{name}_top1_top2_gap"] = compute_top_gap(min_max_normalize(cands))
        feats[f"{name}_score_entropy"] = compute_score_entropy(cands)
    return feats
