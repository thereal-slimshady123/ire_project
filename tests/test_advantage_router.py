"""Tests for the Dual Advantage Regression router, targets, label-stability filter and the CV library."""

import numpy as np
import pandas as pd
import pytest

from router_retrieval.features.feature_builder import FEATURE_COLUMNS
from router_retrieval.router.advantage_targets import (
    advantage_targets, alphas_in_table, label_noise_summary, select_alpha, stability_mask, winner_labels,
)
from router_retrieval.router.cv import (
    FEATURE_GROUPS, cv_predict, fit_predict_fold, predictions_frame, summarize_heldout,
)
from router_retrieval.router.gbm_router import AdvantageRouter, decide_route, route_label, routed_gain
from tests.track2_fixtures import ALPHAS, make_route_table

CFG = {"models": {"gbm": {"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 7, "class_weight": "balanced"},
                  "logistic": {"C": 1.0, "max_iter": 500, "class_weight": "balanced"}},
       "advantage": {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 10}}


# ------------------------------------------------------------------ decision rule & labels

def test_decide_route_rule():
    pred = np.array([[0.10, 0.02],    # bm25 clears theta and beats dense
                     [0.01, 0.12],    # dense clears theta
                     [0.015, 0.01],   # nothing clears theta=0.02 -> hybrid
                     [0.05, 0.05],    # tie: neither strictly larger -> hybrid
                     [-0.1, -0.2]])
    assert decide_route(pred, 0.02, 0.02).tolist() == ["bm25", "dense", "hybrid", "hybrid", "hybrid"]
    assert decide_route(pred, np.inf, np.inf).tolist() == ["hybrid"] * 5
    assert decide_route(pred, 0.0, 0.2).tolist()[1] == "hybrid"  # separate per-route thresholds


def test_route_label_and_routed_gain():
    b, d = np.array([0.3, 0.0, 0.01, -0.2]), np.array([0.1, 0.4, 0.015, -0.1])
    assert route_label(b, d, 0.02).tolist() == ["bm25", "dense", "hybrid", "hybrid"]
    delta = np.column_stack([b, d])
    np.testing.assert_allclose(routed_gain(delta, np.array(["bm25", "dense", "hybrid", "hybrid"], dtype=object)),
                               [0.3, 0.4, 0.0, 0.0])


# ------------------------------------------------------------------------- router fitting

def _planted(n=500, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    db, dd = rng.normal(-0.02, 0.1, n), rng.normal(-0.02, 0.1, n)
    if signal:
        db, dd = db + 0.4 * (X.a > 0.5), dd + 0.4 * (X.b > 0.5)
    return X, db, dd


def test_learns_planted_signal_and_beats_hybrid():
    X, db, dd = _planted(600)
    r = AdvantageRouter(n_estimators=60, learning_rate=0.1, min_child_samples=10).fit(X[:400], db[:400], dd[:400])
    routes = np.array(r.predict(X[400:]))
    gain = routed_gain(np.column_stack([db[400:], dd[400:]]), routes).mean()
    assert gain > 0.03                      # hybrid has gain 0 by definition
    assert r.theta_bm25 < np.inf            # calibration found the signal significant
    assert (routes == "bm25").mean() > 0.1 and (routes == "dense").mean() > 0.1


def test_noise_falls_back_to_always_hybrid():
    """With no signal and negative-mean advantages the guard must keep the router on hybrid."""
    fallbacks = 0
    for seed in range(8):
        X, db, dd = _planted(300, seed, signal=False)
        r = AdvantageRouter(random_state=seed, n_estimators=40, learning_rate=0.1).fit(X, db, dd)
        fallbacks += bool(np.isinf(r.theta_bm25))
        if np.isinf(r.theta_bm25):
            assert set(r.predict(X)) == {"hybrid"}
    assert fallbacks >= 6


def test_probabilities_are_valid_and_consistent_with_ordering():
    X, db, dd = _planted(400)
    r = AdvantageRouter(n_estimators=40, learning_rate=0.1, min_child_samples=10).fit(X, db, dd)
    p = r.predict_proba(X)
    assert p.shape == (len(X), 3) and np.allclose(p.sum(axis=1), 1.0) and (p >= 0).all()
    d = r.predict_delta(X)
    assert d.shape == (len(X), 2)
    # bigger predicted bm25 advantage -> higher P(bm25) (hybrid logit is fixed at 0)
    hi, lo = np.argmax(d[:, 0]), np.argmin(d[:, 0])
    assert p[hi, 0] > p[lo, 0]
    assert r.temperature > 0 and r.calibrated_


def test_eval_mask_keeps_unscored_rows_out_of_calibration():
    X, db, dd = _planted(400)
    mask = np.arange(400) < 200
    # rows outside the mask carry an adversarial advantage for 'bm25'; they must not drive calibration
    db2 = np.where(mask, db, 5.0)
    r = AdvantageRouter(n_estimators=30, learning_rate=0.1).fit(X, db2, dd, eval_mask=mask)
    assert r.oof_gain_ is not None and r.oof_gain_ < 1.0       # calibrated on the 200 natural rows only


def test_tiny_training_set_is_conservative():
    X, db, dd = _planted(10)
    r = AdvantageRouter(n_estimators=5, min_child_samples=2).fit(X.iloc[:1], db[:1], dd[:1])
    assert set(r.predict(X)) == {"hybrid"}


def test_hist_gbm_backend_and_persistence(tmp_path):
    X, db, dd = _planted(300)
    r = AdvantageRouter(backend="hist_gbm", n_estimators=30, learning_rate=0.1, min_child_samples=10).fit(X, db, dd)
    r.save(tmp_path / "r.pkl")
    r2 = AdvantageRouter.load(tmp_path / "r.pkl")
    np.testing.assert_allclose(r.predict_delta(X), r2.predict_delta(X))
    assert r.feature_importances() is None
    with pytest.raises(ValueError):
        AdvantageRouter(backend="nope")


def test_feature_importances_find_the_planted_features():
    X, db, dd = _planted(500)
    r = AdvantageRouter(n_estimators=60, learning_rate=0.1, min_child_samples=10).fit(X, db, dd, calibrate=False)
    imp = r.feature_importances()
    assert imp["delta_bm25"].idxmax() == "a" and imp["delta_dense"].idxmax() == "b"
    np.testing.assert_allclose(imp.sum().to_numpy(), [1.0, 1.0])


# ---------------------------------------------------------- targets & label stability filter

def test_targets_and_alpha_selection():
    t = make_route_table(100)
    assert alphas_in_table(t) == ALPHAS
    assert select_alpha(t) == 0.5
    tg = advantage_targets(t, 0.5)
    np.testing.assert_allclose(tg["delta_bm25"], t["ndcg_bm25"] - t["ndcg_hyb_0.5"])
    np.testing.assert_allclose(tg["delta_dense"], t["ndcg_dense"] - t["ndcg_hyb_0.5"])


def test_stability_mask_drops_flipping_winners():
    idx = ["stable", "flips"]
    t = pd.DataFrame({"ndcg_bm25": [0.8, 0.8], "ndcg_dense": [0.1, 0.1]}, index=idx)
    for a in ALPHAS:
        t[f"ndcg_hyb_{a}"] = [0.5, 0.5]
    # the second query's hybrid catches up with BM25 at alpha=0.6, so its BM25 win does not survive a shift
    t.loc["flips", "ndcg_hyb_0.6"] = 0.8
    keep = stability_mask(t, 0.5, ALPHAS, shift=0.1, epsilon=0.02)
    assert keep.to_dict() == {"stable": True, "flips": False}
    assert winner_labels(t, 0.5).tolist() == ["bm25", "bm25"]
    s = label_noise_summary(t, 0.5, ALPHAS)
    assert s["unstable_share"] == 0.5 and s["mix_stable"]["bm25"] == 1.0
    # alpha at the grid edge only has one neighbour; must not raise
    assert stability_mask(t, 1.0, ALPHAS).all()


# ------------------------------------------------------------------------------- CV library

def test_cv_covers_every_query_once_and_exposes_contract3():
    t = make_route_table(150)
    held, infos = cv_predict(t, "advantage", FEATURE_COLUMNS, ALPHAS, CFG, k_folds=5, seed=1)
    assert sorted(held.index) == sorted(t.index) and held.index.is_unique
    assert held["fold"].nunique() == 5 and len(infos) == 5
    pred = predictions_frame(held)
    assert list(pred.columns) == ["query_id", "fold", "route_predicted", "p_bm25", "p_dense", "p_hybrid",
                                  "delta_pred_bm25", "delta_pred_dense"]
    assert pred["fold"].dtype == np.int32 and pred["p_dense"].dtype == np.float32 and pred["query_id"].dtype == object
    assert set(pred["route_predicted"]) <= {"bm25", "dense", "hybrid"}
    np.testing.assert_allclose(pred[["p_bm25", "p_dense", "p_hybrid"]].sum(axis=1), 1.0, atol=1e-5)


def test_heldout_routes_ignore_heldout_outcomes():
    """Leakage guard: the routes chosen for held-out queries must not depend on their nDCG columns."""
    t = make_route_table(160, seed=3)
    train, test = t.iloc[:120], t.iloc[120:].copy()
    for kind in ("advantage", "advantage_fixed", "gbm_cls", "logistic_cls"):
        out1, _ = fit_predict_fold(train, test, kind, FEATURE_COLUMNS, ALPHAS, CFG, seed=0)
        scrambled = test.copy()
        rng = np.random.default_rng(9)
        for c in [c for c in scrambled.columns if c.startswith("ndcg_")]:
            scrambled[c] = rng.random(len(scrambled))
        out2, _ = fit_predict_fold(train, scrambled, kind, FEATURE_COLUMNS, ALPHAS, CFG, seed=0)
        assert (out1["route_predicted"] == out2["route_predicted"]).all(), kind


def test_router_beats_hybrid_end_to_end_on_planted_table():
    t = make_route_table(400, seed=5)
    held, _ = cv_predict(t, "advantage", FEATURE_COLUMNS, ALPHAS, CFG, k_folds=4, seed=0)
    s = summarize_heldout(held, n_bootstrap=500, seed=0)
    ci = s["tests"]["routed - hybrid"]
    assert ci["delta_mean"] > 0.03 and ci["ci_95_low"] > 0
    assert s["mean"]["oracle"] >= s["mean"]["routed"] >= 0
    assert 0 < s["headroom_captured"] <= 1.0 + 1e-9
    assert sum(s["wins_losses_ties"].values()) == s["n"]


def test_synthetic_rows_join_training_but_never_the_test_side():
    t = make_route_table(160, seed=2)
    syn = make_route_table(200, seed=7, name="syn")
    syn.index = [f"s{i}" for i in range(len(syn))]
    out, info = fit_predict_fold(t.iloc[:120], t.iloc[120:], "advantage", FEATURE_COLUMNS, ALPHAS, CFG, seed=0,
                                 synthetic=syn, synthetic_weight=0.5)
    assert info["n_synthetic"] == 200 and info["n_fit_rows"] == 320
    assert list(out.index) == list(t.index[120:])            # only natural queries are scored
    with pytest.raises(ValueError):
        fit_predict_fold(t.iloc[:120], t.iloc[120:], "gbm_cls", FEATURE_COLUMNS, ALPHAS, CFG, synthetic=syn)


def test_stability_filter_reduces_fit_rows():
    t = make_route_table(200, seed=4)
    _, plain = fit_predict_fold(t.iloc[:150], t.iloc[150:], "advantage", FEATURE_COLUMNS, ALPHAS, CFG)
    _, filt = fit_predict_fold(t.iloc[:150], t.iloc[150:], "advantage", FEATURE_COLUMNS, ALPHAS, CFG, stability=True)
    assert filt["n_fit_rows"] <= plain["n_fit_rows"] and "stability_dropped" in filt


def test_feature_groups_partition_the_feature_space():
    assert set(FEATURE_GROUPS["query_only"]) | set(FEATURE_GROUPS["retrieval_signal"]) == set(FEATURE_COLUMNS)
    assert not set(FEATURE_GROUPS["query_only"]) & set(FEATURE_GROUPS["retrieval_signal"])
    assert all(f.startswith("bm25_") or f in FEATURE_GROUPS["query_only"] for f in FEATURE_GROUPS["pre_dense"])
