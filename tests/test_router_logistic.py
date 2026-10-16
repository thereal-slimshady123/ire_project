"""Unit tests for Logistic Regression router."""

import numpy as np
import pandas as pd

from router_retrieval.router.logistic_router import LogisticRouter


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(np.where(X.f1 > 0.5, "bm25", np.where(X.f1 < -0.5, "dense", "hybrid")))
    return X, y


def test_logistic_router_fit_predict():
    """Test logistic router trains and outputs valid probabilities summing to 1."""
    X, y = _data()
    r = LogisticRouter().fit(X, y)
    p = r.predict_proba(X)
    assert p.shape == (len(X), 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (np.array(r.predict(X)) == y.values).mean() > 0.8


def test_logistic_router_class_weight():
    """Test balanced class weights handle imbalanced class labels."""
    X, y = _data()
    keep = (y != "bm25") | (np.arange(len(y)) % 10 == 0)  # make bm25 rare
    r = LogisticRouter(class_weight="balanced").fit(X[keep], y[keep])
    assert "bm25" in r.predict(X[y == "bm25"])


def test_missing_class_gets_zero_probability():
    X, y = _data()
    keep = y != "dense"
    p = LogisticRouter().fit(X[keep], y[keep]).predict_proba(X)
    assert np.allclose(p[:, 1], 0.0)  # column order follows classes: bm25, dense, hybrid
