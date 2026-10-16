"""Unit tests for retrieval signal features."""

import pytest

from router_retrieval.data.schema import Candidate
from router_retrieval.features.retrieval_signal_features import (
    compute_jaccard_overlap,
    compute_query_term_coverage,
    compute_rank_correlation,
    compute_score_entropy,
    compute_top_gap,
)


def _cands(ids, scores=None):
    scores = scores or [float(len(ids) - i) for i in range(len(ids))]
    return [Candidate("q", d, s, i + 1) for i, (d, s) in enumerate(zip(ids, scores))]


def test_jaccard_overlap():
    """Test Jaccard overlap on fully overlapping, partially overlapping, and disjoint sets."""
    assert compute_jaccard_overlap(_cands("abc"), _cands("cba"), k=3) == 1.0
    assert compute_jaccard_overlap(_cands("abc"), _cands("bcd"), k=3) == pytest.approx(2 / 4)
    assert compute_jaccard_overlap(_cands("abc"), _cands("xyz"), k=3) == 0.0


def test_rank_correlation():
    """Test Spearman and Kendall rank correlations on known ranking lists."""
    assert compute_rank_correlation(_cands("abcd"), _cands("abcd")) == pytest.approx(1.0)
    assert compute_rank_correlation(_cands("abcd"), _cands("dcba"), method="kendall") == pytest.approx(-1.0)
    assert compute_rank_correlation(_cands("abcd"), _cands("axyz")) == 0.0  # < 3 shared


def test_score_entropy():
    """Test Shannon entropy calculation on uniform vs peaked distributions."""
    peaked = compute_score_entropy(_cands("abcde", [10.0, 0.1, 0.1, 0.1, 0.0]), k=5)
    flat = compute_score_entropy(_cands("abcde", [1.0, 0.99, 0.98, 0.97, 0.0]), k=5)
    assert 0.0 <= peaked < flat <= 1.0


def test_top_score_gap():
    """Test top1 - top2 margin score calculation."""
    assert compute_top_gap(_cands("ab", [5.0, 3.5])) == pytest.approx(1.5)
    assert compute_top_gap(_cands("a", [5.0])) == 0.0


def test_query_term_coverage():
    assert compute_query_term_coverage(["a", "b"], [{"a", "b"}, {"a"}, set()]) == pytest.approx(0.5)
