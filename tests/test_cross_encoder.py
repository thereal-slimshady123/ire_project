"""Unit tests for cross-encoder reranker."""

import pytest

# Placeholders: remove this marker once the module under test is implemented.
pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_cross_encoder_rerank_pool():
    """Test the top rerank_depth hybrid-ordered candidates are reordered by cross-encoder score."""
    pass


def test_cross_encoder_appends_tail_to_output_depth():
    """Test candidates beyond rerank_depth are appended in hybrid order, up to output_depth."""
    pass


def test_cross_encoder_scores_strictly_decreasing():
    """Test returned scores strictly decrease with rank (pytrec_eval sorts by score)."""
    pass
