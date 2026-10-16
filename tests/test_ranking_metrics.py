"""Unit tests for ranking metrics computation."""

import math

import numpy as np
import pytest

from router_retrieval.evaluation.ranking_metrics import compute_ranking_metrics

QRELS = {"q1": {"d1": 2, "d2": 1, "d9": 1}, "q2": {"d5": 1}}


def test_ndcg_at_10_perfect_ranking():
    """Test nDCG@10 evaluates to 1.0 for perfect ranking."""
    df = compute_ranking_metrics(QRELS, {"q1": ["d1", "d2", "d9"], "q2": ["d5"]}, ["ndcg_cut_10"])
    assert df["ndcg_cut_10"].tolist() == [pytest.approx(1.0), pytest.approx(1.0)]


def test_ndcg_graded_hand_computed():
    df = compute_ranking_metrics(QRELS, {"q1": ["d2", "x", "d1"]}, ["ndcg_cut_10"])
    dcg = 1 / math.log2(2) + 2 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert df.loc["q1", "ndcg_cut_10"] == pytest.approx(dcg / idcg)


def test_mrr_first_rank():
    """Test MRR evaluates to 1.0 when relevant item is top ranked."""
    df = compute_ranking_metrics(QRELS, {"q1": ["d1", "x"], "q2": ["x", "y", "d5"]}, ["recip_rank", "mrr_10"])
    assert df.loc["q1", "recip_rank"] == 1.0
    assert df.loc["q2", "mrr_10"] == pytest.approx(1 / 3)


def test_empty_retrieval_metrics():
    """Test metrics evaluate cleanly to 0.0 when no relevant items retrieved."""
    df = compute_ranking_metrics(QRELS, {"q1": ["x", "y"]})
    assert (df.loc["q2"] == 0.0).all()  # query missing from run entirely
    assert (df.loc["q1"] == 0.0).all()


def test_matches_pytrec_eval():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = np.random.default_rng(0)
    docs = [f"d{i}" for i in range(150)]
    qrels = {
        f"q{j}": {str(d): int(rng.integers(0, 3)) for d in rng.choice(docs, 20, replace=False)} for j in range(20)
    }
    run = {q: [str(d) for d in rng.permutation(docs)[:120]] for q in qrels}
    metrics = ["ndcg_cut_10", "recip_rank", "map", "recall_100"]
    ours = compute_ranking_metrics(qrels, run, metrics)
    scored_run = {q: {d: float(len(ds) - i) for i, d in enumerate(ds)} for q, ds in run.items()}
    ref = pytrec_eval.RelevanceEvaluator(qrels, set(metrics)).evaluate(scored_run)
    for q in qrels:
        for m in metrics:
            assert ours.loc[q, m] == pytest.approx(ref[q][m], abs=1e-6), (q, m)
