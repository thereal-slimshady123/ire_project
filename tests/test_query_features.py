"""Unit tests for query-intrinsic features."""

import pytest

from router_retrieval.data.schema import Query
from router_retrieval.features.query_features import extract_query_features

IDF = {"vitamin": 2.0, "d": 1.0, "bone": 4.0}


def test_query_length_extraction():
    """Test token length feature computation."""
    f = extract_query_features(Query("q", "Is vitamin D good for the bones?"), IDF)
    assert f["query_length"] == 7


def test_idf_features():
    """Test max and mean IDF features against known vocabulary dictionary."""
    f = extract_query_features(Query("q", "vitamin D bone"), IDF, rare_threshold=3.0)
    assert f["max_idf"] == pytest.approx(4.0)
    assert f["mean_idf"] == pytest.approx(7.0 / 3)
    assert f["rare_term_density"] == pytest.approx(1 / 3)


def test_oov_fraction():
    """Test out-of-vocabulary fraction calculation."""
    f = extract_query_features(Query("q", "vitamin xylophone"), IDF)
    assert f["oov_fraction"] == pytest.approx(0.5)
    assert f["max_idf"] == pytest.approx(2.0)


def test_entity_density():
    f = extract_query_features(Query("q", "effects of COVID-19 in Italy"), IDF)
    # content tokens: effects, COVID, 19, Italy -> COVID (acronym), 19 (digit), Italy (mid-query cap)
    assert f["entity_density"] == pytest.approx(3 / 4)
