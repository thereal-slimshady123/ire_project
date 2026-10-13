"""Unit tests for paired bootstrap significance testing."""

import pytest

# Placeholders: remove this marker once the module under test is implemented.
pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_significance_identical_systems():
    """Test paired bootstrap between identical systems yields p-value near 1.0 and delta 0."""
    pass


def test_significance_clearly_different_systems():
    """Test paired bootstrap between strictly superior system yields significant p-value."""
    pass


def test_holm_bonferroni_adjustment():
    """Test Holm-Bonferroni adjusted p-values on a known example."""
    pass
