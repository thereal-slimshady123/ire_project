"""Unit tests for score normalization."""

import numpy as np
import pytest

from router_retrieval.data.schema import Candidate
from router_retrieval.retrieval.normalization import normalize_scores


def _cands(scores):
    return [Candidate("q", f"d{i}", s, i + 1) for i, s in enumerate(scores)]


def test_min_max_normalization_range():
    """Test min-max normalization bounds scores to [0, 1]."""
    out = [c.score for c in normalize_scores(_cands([12.0, 7.5, 3.0]), "min_max")]
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(0.0)


def test_min_max_identical_scores():
    """Test edge case where all scores are identical."""
    out = normalize_scores(_cands([0.8, 0.8]), "min_max")
    assert [c.score for c in out] == [0.0, 0.0]
    assert normalize_scores([], "min_max") == []


def test_min_max_floor_anchor():
    out = [c.score for c in normalize_scores(_cands([8.0]), "min_max", floor=0.0)]
    assert out == [pytest.approx(1.0)]
    out = [c.score for c in normalize_scores(_cands([10.0, 5.0]), "min_max", floor=0.0)]
    assert out == [pytest.approx(1.0), pytest.approx(0.5)]


def test_z_score_normalization():
    """Test z-score normalization yields zero mean and unit variance."""
    out = np.array([c.score for c in normalize_scores(_cands([1.0, 2.0, 3.0, 10.0]), "z_score")])
    assert out.mean() == pytest.approx(0.0, abs=1e-9)
    assert out.std() == pytest.approx(1.0, abs=1e-6)


def test_normalization_does_not_mutate_input():
    cands = _cands([5.0, 1.0])
    normalize_scores(cands)
    assert [c.score for c in cands] == [5.0, 1.0]
