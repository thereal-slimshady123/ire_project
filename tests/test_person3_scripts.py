"""Unit tests for Person 3's evaluation scripts (tau escalation, policy latency, labels, headroom)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tau = _load("tau_escalation", "run_tau_escalation.py")
latency = _load("policy_latency", "run_policy_latency.py")
labels = _load("generate_labels", "generate_labels.py")
headroom = _load("oracle_headroom", "run_oracle_headroom.py")
full = _load("full_evaluation", "run_full_evaluation.py")


def test_tau_policy_scores_and_random_control_use_realized_rate():
    out = tau.apply_tau_policy(routed=[0.2, 0.4, 0.6, 0.8], hybrid=[0.5] * 4, reranked=[1.0, 0.0, 1.0, 0.0],
                               escalated=[True, False, False, False])
    assert out["router_tau"].tolist() == [1.0, 0.4, 0.6, 0.8]
    assert out["hybrid_tau"].tolist() == [1.0, 0.5, 0.5, 0.5]
    # rate 25%: 0.75 * routed + 0.25 * reranked
    assert out["router_random"] == pytest.approx([0.4, 0.3, 0.7, 0.6])
    assert out["hybrid_random"] == pytest.approx([0.625, 0.375, 0.625, 0.375])


def test_aurc_rewards_confidence_that_ranks_risky_queries_last():
    risk = np.array([0.0, 0.0, 1.0])
    good, bad = np.array([0.9, 0.8, 0.1]), np.array([0.1, 0.2, 0.9])
    assert tau.aurc(good, risk) == pytest.approx((0 + 0 + 1 / 3) / 3)
    assert tau.aurc(bad, risk) == pytest.approx((1 + 0.5 + 1 / 3) / 3)
    assert tau.aurc(good, risk) < risk.mean() < tau.aurc(bad, risk)


def test_constant_router_is_fully_confident():
    class Constant:
        constant = "hybrid"
    p = tau.router_proba(Constant(), pd.DataFrame({"f": [1.0, 2.0]}))
    assert p.tolist() == [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]


def test_escalation_masks_follow_experiment4_rule_per_fold():
    frame = pd.DataFrame({
        "dataset": ["a"] * 4 + ["b"] * 2, "fold": [0, 0, 0, 0, 0, 0],
        "score_predicted_gain": [0.1, 0.9, 0.5, 0.2, 0.3, 0.7],
        "score_router_confidence": [0.0] * 6, "score_oracle": [1, 2, 3, 4, 5, 6],
    })
    masks = latency.escalation_masks(frame, 0.5, seed=0)
    assert masks["predicted_gain"].tolist() == [False, True, True, False, False, True]
    assert masks["router_confidence"].tolist() == [True, True, False, False, True, False]  # stable on ties
    assert masks["oracle"].tolist() == [False, False, True, True, False, True]
    assert masks["random"][:4].sum() == 2 and masks["random"][4:].sum() == 1


def test_policy_latency_adds_cross_encoder_only_when_escalated():
    c = pd.DataFrame({name: [1.0, 1.0] for name in latency.COMPONENTS})
    c["cross_encoder_rerank"] = [100.0, 100.0]
    assert latency.policy_latency(c, "always_hybrid").tolist() == [4.0, 4.0]
    assert latency.policy_latency(c, "always_rerank").tolist() == [104.0, 104.0]
    assert latency.policy_latency(c, "router_tau", escalated=[True, False]).tolist() == [106.0, 6.0]
    assert latency.policy_latency(c, "predicted_gain", escalated=[False, True]).tolist() == [6.0, 106.0]
    # skip-dense pays BM25 + skip features + router, plus dense/fusion only when not skipped
    assert latency.policy_latency(c, "skip_dense", skipped=[True, False]).tolist() == [3.0, 6.0]
    with pytest.raises(ValueError):
        latency.policy_latency(c, "nope")


def test_fold_labels_use_each_folds_alpha_and_epsilon():
    table = pd.DataFrame({"ndcg_bm25": [0.5, 0.5, 0.5, 0.5], "ndcg_dense": [0.0] * 4,
                          "ndcg_hyb_0.4": [0.49] * 4, "ndcg_hyb_0.6": [0.2] * 4},
                         index=["q1", "q2", "q3", "q4"])
    info = [{"alpha": 0.4, "epsilon": 0.02}, {"alpha": 0.6, "epsilon": 0.02}]
    out = labels.fold_labels(table, info, k_folds=2, seed=42)
    assert list(out.index) == ["q1", "q2", "q3", "q4"]
    for q, row in out.iterrows():
        # alpha 0.4: BM25 leads hybrid by 0.01 < eps -> hybrid; alpha 0.6: BM25 leads by 0.3 -> bm25
        assert row["best_route"] == ("hybrid" if row["alpha"] == 0.4 else "bm25")
        assert row["max_regret"] == pytest.approx(0.5)


def test_noise_oracle_uses_grid_neighbours_and_clips_at_edges():
    grid = [0.0, 0.1, 0.2]
    table = pd.DataFrame({"ndcg_bm25": [0.0, 0.0], "ndcg_dense": [0.0, 0.0], "ndcg_hyb_0.0": [0.1, 0.9],
                          "ndcg_hyb_0.1": [0.3, 0.5], "ndcg_hyb_0.2": [0.2, 0.1]}, index=["a", "b"])
    f = headroom.headroom_frame(table, pd.Series({"a": 0.1, "b": 0.2}), grid)
    assert f.loc["a", "noise"] == 0.3 and f.loc["a", "hybrid"] == 0.3
    assert f.loc["b", "noise"] == 0.5 and f.loc["b", "hybrid"] == 0.1  # alpha 0.2 only has neighbour 0.1


def test_mean_ci_brackets_the_mean():
    out = full._mean_ci([0.0, 1.0, 0.5, 0.5], n_bootstrap=500)
    assert out["mean"] == pytest.approx(0.5)
    assert out["ci_95"][0] <= 0.5 <= out["ci_95"][1]
