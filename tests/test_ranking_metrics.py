"""Unit tests for ranking metrics computation."""

import pytest

# Placeholders: remove this marker once the module under test is implemented.
pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_ndcg_at_10_perfect_ranking():
    """Test nDCG@10 evaluates to 1.0 for perfect ranking."""
    pass


def test_mrr_first_rank():
    """Test MRR evaluates to 1.0 when relevant item is top ranked."""
    pass


def test_metrics_use_rank_order_not_raw_scores():
    """Test a run with non-monotone raw scores is evaluated in list (rank) order."""
    pass


def test_mrr_at_10_cutoff():
    """Test MRR@10 is 0.0 when the first relevant document is at rank 11."""
    pass


def test_empty_retrieval_metrics():
    """Test metrics evaluate cleanly to 0.0 when no relevant items retrieved."""
    pass
