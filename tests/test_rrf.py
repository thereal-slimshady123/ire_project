"""Unit tests for Reciprocal Rank Fusion."""

import pytest


from router_retrieval.data.schema import Candidate
from router_retrieval.retrieval.rrf import ReciprocalRankFusion


def test_rrf_scoring_order():
    """Test reciprocal rank fusion ranking order on known list inputs."""
    rrf = ReciprocalRankFusion(k=60)
    sys1 = [
        Candidate(query_id="q1", doc_id="d1", score=10.0, rank=1),
        Candidate(query_id="q1", doc_id="d2", score=5.0, rank=2),
        Candidate(query_id="q1", doc_id="d3", score=1.0, rank=3),
    ]
    sys2 = [
        Candidate(query_id="q1", doc_id="d2", score=0.9, rank=1),
        Candidate(query_id="q1", doc_id="d1", score=0.8, rank=2),
        Candidate(query_id="q1", doc_id="d4", score=0.7, rank=3),
    ]

    fused = rrf.fuse([sys1, sys2], top_k=10)
    assert len(fused) == 4
    # d1 and d2 have identical sum of 1/61 + 1/62; d1 < d2 tiebreak
    assert fused[0].doc_id == "d1"
    assert fused[1].doc_id == "d2"
    expected_top = 1.0 / 61.0 + 1.0 / 62.0
    assert fused[0].score == pytest.approx(expected_top)
    assert fused[1].score == pytest.approx(expected_top)
    assert [c.rank for c in fused] == [1, 2, 3, 4]
    # d3 and d4 both appear once at rank 3: 1/63; d3 < d4 tiebreak
    assert fused[2].doc_id == "d3"
    assert fused[3].doc_id == "d4"
    assert fused[2].score == pytest.approx(1.0 / 63.0)


def test_rrf_disjoint_lists():
    """Test RRF fusion when candidate lists are completely disjoint."""
    rrf = ReciprocalRankFusion(k=60)
    sys1 = [
        Candidate(query_id="q1", doc_id="d1", score=10.0, rank=1),
        Candidate(query_id="q1", doc_id="d2", score=5.0, rank=2),
    ]
    sys2 = [
        Candidate(query_id="q1", doc_id="d3", score=0.9, rank=1),
        Candidate(query_id="q1", doc_id="d4", score=0.8, rank=2),
    ]
    fused = rrf.fuse([sys1, sys2], top_k=10)
    assert [c.doc_id for c in fused] == ["d1", "d3", "d2", "d4"]
    assert [c.rank for c in fused] == [1, 2, 3, 4]


def test_rrf_empty_and_truncation():
    """Test RRF with empty lists and top_k truncation."""
    rrf = ReciprocalRankFusion(k=60)
    assert rrf.fuse([]) == []
    assert rrf.fuse([[], []]) == []

    sys1 = [Candidate("q", f"d{i}", float(10 - i), i + 1) for i in range(10)]
    fused = rrf.fuse([sys1], top_k=3)
    assert len(fused) == 3
    assert [c.rank for c in fused] == [1, 2, 3]
    assert [c.doc_id for c in fused] == ["d0", "d1", "d2"]

