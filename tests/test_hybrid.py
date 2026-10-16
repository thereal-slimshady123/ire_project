"""Unit tests for fixed-alpha hybrid retriever."""

import pytest

from router_retrieval.data.schema import Candidate
from router_retrieval.retrieval.hybrid import HybridRetriever

BM25 = [Candidate("q", "a", 10.0, 1), Candidate("q", "b", 6.0, 2), Candidate("q", "c", 2.0, 3)]
DENSE = [Candidate("q", "c", 0.9, 1), Candidate("q", "d", 0.7, 2), Candidate("q", "a", 0.5, 3)]


def _hybrid(alpha):
    return HybridRetriever(bm25_retriever=None, dense_retriever=None, alpha=alpha)


def test_hybrid_convex_combination_alpha_zero():
    """Test alpha=0 reduces exactly to normalized BM25 ranking."""
    out = _hybrid(0.0).combine_candidates(BM25, DENSE, k=3)
    assert [c.doc_id for c in out] == ["a", "b", "c"]


def test_hybrid_convex_combination_alpha_one():
    """Test alpha=1 reduces exactly to normalized Dense ranking."""
    out = _hybrid(1.0).combine_candidates(BM25, DENSE, k=3)
    assert [c.doc_id for c in out] == ["c", "d", "a"]


def test_hybrid_intermediate_alpha():
    """Test convex combination score calculation for intermediate alpha."""
    # k=3: both lists are full-length, so min-max is anchored at each list's minimum.
    out = {c.doc_id: c.score for c in _hybrid(0.5).combine_candidates(BM25, DENSE, k=3)}
    # bm25 norm: a=1, b=0.5, c=0 ; dense norm: c=1, d=0.5, a=0 ; missing -> 0 (b, d tie -> truncated)
    assert out["a"] == pytest.approx(0.5)
    assert out["c"] == pytest.approx(0.5)
    assert len(out) == 3


def test_hybrid_truncated_bm25_list_anchors_at_zero():
    # k=10: the 3-doc BM25 list is truncated, so unretrieved docs have BM25 score 0 -> floor 0.
    out = {c.doc_id: c.score for c in _hybrid(0.5).combine_candidates(BM25, DENSE, k=10)}
    # bm25 norm: a=1, b=0.6, c=0.2 ; dense (also short, but no known floor) norm: c=1, d=0.5, a=0
    assert out["a"] == pytest.approx(0.5)
    assert out["b"] == pytest.approx(0.3)
    assert out["c"] == pytest.approx(0.6)
    assert out["d"] == pytest.approx(0.25)


def test_short_bm25_list_alpha_extremes_reproduce_single_retrievers():
    """Regression: a 1-2 doc BM25 list must not normalize its hits down to the 'unretrieved' score."""
    dense = [Candidate("q", f"z{i}", 0.9 - 0.01 * i, i + 1) for i in range(5)]  # doc ids sort before 'm*'
    for bm25 in ([Candidate("q", "m1", 9.0, 1)], [Candidate("q", "m1", 14.0, 1), Candidate("q", "m2", 12.5, 2)]):
        h0 = HybridRetriever(None, None, alpha=0.0).combine_candidates(bm25, dense, k=5)
        assert [c.doc_id for c in h0[: len(bm25)]] == [c.doc_id for c in bm25]
        h1 = HybridRetriever(None, None, alpha=1.0).combine_candidates(bm25, dense, k=5)
        assert [c.doc_id for c in h1] == [c.doc_id for c in dense]


def test_empty_bm25_list_falls_back_to_dense_order_at_alpha_zero():
    dense = [Candidate("q", d, s, i + 1) for i, (d, s) in enumerate([("c", 0.9), ("a", 0.5), ("b", 0.1)])]
    out = HybridRetriever(None, None, alpha=0.0).combine_candidates([], dense, k=3)
    assert [c.doc_id for c in out] == ["c", "a", "b"]


def test_hybrid_ranks_are_contiguous_and_truncated():
    out = _hybrid(0.3).combine_candidates(BM25, DENSE, k=2)
    assert [c.rank for c in out] == [1, 2]


def test_hybrid_rejects_bad_alpha():
    with pytest.raises(ValueError):
        _hybrid(1.5)
