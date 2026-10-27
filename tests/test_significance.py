"""Unit tests for paired bootstrap significance testing."""

import numpy as np
import pytest

from router_retrieval.evaluation.significance import paired_bootstrap_test, paired_wilcoxon_test


def test_significance_identical_systems():
    """Test paired bootstrap between identical systems yields p-value near 1.0 and delta 0."""
    a = np.random.default_rng(1).random(200)
    res = paired_bootstrap_test(a, a, n_resamples=2000)
    assert res["delta_mean"] == 0.0
    assert res["p_value"] == pytest.approx(1.0)
    assert paired_wilcoxon_test(a, a)["p_value"] == 1.0


def test_significance_clearly_different_systems():
    """Test paired bootstrap between strictly superior system yields significant p-value."""
    rng = np.random.default_rng(2)
    b = rng.random(200)
    a = b + 0.05 + rng.normal(0, 0.01, 200)
    res = paired_bootstrap_test(a, b, n_resamples=2000)
    assert res["delta_mean"] == pytest.approx(0.05, abs=0.01)
    assert res["p_value"] < 0.001
    assert res["ci_95_low"] > 0
    assert paired_wilcoxon_test(a, b, alternative="greater")["p_value"] < 0.001


def test_significance_is_seeded():
    rng = np.random.default_rng(3)
    a, b = rng.random(50), rng.random(50)
    assert paired_bootstrap_test(a, b, 1000, seed=7) == paired_bootstrap_test(a, b, 1000, seed=7)


def test_ndcg_uses_trec_eval_graded_gain():
    from router_retrieval.evaluation.ranking_metrics import ndcg_at_k
    actual = ndcg_at_k(["rel2", "rel1"], {"rel2": 2, "rel1": 1}, 2)
    ideal = (3 + 1 / np.log2(3)) / (3 + 1 / np.log2(3))
    assert actual == pytest.approx(ideal)


def test_wilcoxon_rejects_bad_shapes():
    with pytest.raises(ValueError):
        paired_wilcoxon_test([1, 2], [1])
