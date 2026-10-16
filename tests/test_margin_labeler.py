"""Unit tests for margin-based labeler."""

import pandas as pd
import pytest

from router_retrieval.labeling.margin_labeler import MarginLabeler
from router_retrieval.labeling.regret import compute_regret_weights


def _wide(rows):
    return pd.DataFrame(rows, columns=["bm25", "dense", "hybrid"], index=[f"q{i}" for i in range(len(rows))])


def test_clear_winner_label():
    """Test when one route clearly beats others beyond epsilon."""
    out = MarginLabeler(0.02).label_wide(_wide([[0.9, 0.5, 0.6], [0.1, 0.8, 0.5]]))
    assert out["best_route"].tolist() == ["bm25", "dense"]
    assert out["margin"].tolist() == [pytest.approx(0.3), pytest.approx(0.3)]


def test_margin_tie_defaults_to_hybrid():
    """Test when top two routes are within epsilon, defaulting to hybrid."""
    out = MarginLabeler(0.05).label_wide(_wide([[0.60, 0.58, 0.10]]))
    assert out["best_route"].iloc[0] == "hybrid"


def test_exact_tie():
    """Test exact tie edge case."""
    out = MarginLabeler(0.0).label_wide(_wide([[0.5, 0.5, 0.5], [0.7, 0.2, 0.7]]))
    assert out["best_route"].tolist() == ["hybrid", "hybrid"]


def test_tune_epsilon_uses_utility_and_prefers_smaller_on_ties():
    lab = MarginLabeler(0.02)
    assert lab.tune_epsilon(None, [0.0, 0.05, 0.1], utility_fn=lambda e: {0.0: 0.4, 0.05: 0.5, 0.1: 0.5}[e]) == 0.05


def test_long_format_and_regret():
    long = pd.DataFrame({
        "query_id": ["a"] * 3, "route": ["bm25", "dense", "hybrid"], "ndcg_cut_10": [0.2, 0.9, 0.5],
    })
    assert MarginLabeler(0.02).label_queries(long)["best_route"].iloc[0] == "dense"
    reg = compute_regret_weights(long).iloc[0]
    assert reg["regret_bm25"] == pytest.approx(0.7)
    assert reg["regret_dense"] == pytest.approx(0.0)
    assert reg["max_regret"] == pytest.approx(0.7)
