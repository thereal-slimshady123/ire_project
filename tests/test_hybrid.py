"""Unit tests for fixed-alpha hybrid retriever."""

import pytest

# Placeholders: remove this marker once the module under test is implemented.
pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_hybrid_convex_combination_alpha_zero():
    """Test alpha=0 reduces exactly to normalized BM25 ranking."""
    pass


def test_hybrid_convex_combination_alpha_one():
    """Test alpha=1 reduces exactly to normalized Dense ranking."""
    pass


def test_hybrid_union_scored_exactly():
    """Test docs returned by only one retriever get their exact score from the other retriever."""
    pass


def test_hybrid_intermediate_alpha():
    """Test convex combination score calculation for intermediate alpha."""
    pass
