"""Unit tests for cross-encoder reranker."""

import pytest


def test_cross_encoder_rerank_pool():
    """Test cross-encoder reranks top candidate pool and updates scores."""
    from router_retrieval.data.schema import Candidate, Document, Query
    from router_retrieval.retrieval.cross_encoder import CrossEncoderReranker
    r = CrossEncoderReranker(candidate_pool_size=2)
    r.score_pairs = lambda pairs, show_progress_bar=False: [0.1, 0.9]
    c = [Candidate("q", "a", 3, 1), Candidate("q", "b", 2, 2), Candidate("q", "c", 1, 3)]
    out = r.rerank(Query("q", "query"), c, {x: Document(x, x) for x in "abc"}, top_k=3)
    assert [x.doc_id for x in out] == ["b", "a", "c"]
    assert [x.score for x in out] == [0.9, 0.1, 1.0]


def test_cross_encoder_truncation():
    """Test candidate pool size limit truncates correctly."""
    from router_retrieval.data.schema import Candidate, Document, Query
    from router_retrieval.retrieval.cross_encoder import CrossEncoderReranker
    r = CrossEncoderReranker(candidate_pool_size=1)
    r.score_pairs = lambda pairs, show_progress_bar=False: [2.0]
    c = [Candidate("q", "a", 3, 1), Candidate("q", "b", 2, 2)]
    out = r.rerank(Query("q", "query"), c, {x: Document(x, x) for x in "ab"}, top_k=10)
    assert [x.doc_id for x in out] == ["a", "b"]
