"""Unit tests for GBM router."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
from router_retrieval.router.gbm_router import GBMRouter  # noqa: E402


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(np.where(X.f1 > 0.5, "bm25", np.where(X.f1 < -0.5, "dense", "hybrid")))
    return X, y


def test_gbm_router_fit_predict():
    """Test GBM router trains and outputs valid class probabilities."""
    X, y = _data()
    r = GBMRouter(n_estimators=50).fit(X, y)
    p = r.predict_proba(X)
    assert p.shape == (len(X), 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (np.array(r.predict(X)) == y.values).mean() > 0.9


def test_gbm_router_regret_weighting():
    """Test sample weight incorporation with regret values."""
    X, y = _data()
    w = np.where(y == "bm25", 50.0, 1.0)
    p_plain = GBMRouter(n_estimators=50, class_weight=None).fit(X, y).predict_proba(X)[:, 0].mean()
    p_weighted = GBMRouter(n_estimators=50, class_weight=None).fit(X, y, sample_weight=w).predict_proba(X)[:, 0].mean()
    assert p_weighted > p_plain


def test_experiment2_heldout_routes_ignore_test_relevance():
    """Leakage guard: routes chosen for test queries must not depend on the test queries' scores."""
    spec = importlib.util.spec_from_file_location(
        "exp2", Path(__file__).resolve().parents[1] / "scripts" / "run_experiment2_heldout_routing.py"
    )
    exp2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp2)

    rng = np.random.default_rng(1)
    n, alphas = 120, [0.0, 0.5, 1.0]
    df = pd.DataFrame({f: rng.normal(size=n) for f in exp2.FEATURES}, index=[f"q{i}" for i in range(n)])
    df["dataset"] = "toy"
    df["ndcg_bm25"] = rng.random(n)
    df["ndcg_dense"] = rng.random(n)
    for a in alphas:
        df[f"ndcg_hyb_{a}"] = rng.random(n)
    cfg = {"models": {"gbm": {"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 7, "class_weight": "balanced"},
                      "logistic": {"C": 1.0, "max_iter": 500, "class_weight": "balanced"}}}
    train, test = df.iloc[:90], df.iloc[90:].copy()
    out1, _ = exp2.fit_and_evaluate(train, test, alphas, [0.0, 0.05], cfg, seed=0)
    for c in ["ndcg_bm25", "ndcg_dense", *[f"ndcg_hyb_{a}" for a in alphas]]:
        test[c] = rng.random(len(test))
    out2, _ = exp2.fit_and_evaluate(train, test, alphas, [0.0, 0.05], cfg, seed=0)
    assert (out1["route_gbm"] == out2["route_gbm"]).all()
    assert (out1["route_logistic"] == out2["route_logistic"]).all()
