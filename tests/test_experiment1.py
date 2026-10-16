"""Unit tests for Experiment 1 partitioning and alpha selection logic."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "exp1", Path(__file__).resolve().parents[1] / "scripts" / "run_experiment1_hybrid_degradation.py"
)
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)


@pytest.mark.parametrize(
    "b,d,h,expected",
    [
        (0.8, 0.5, 0.6, "bm25_dominant"),
        (0.2, 0.7, 0.5, "dense_dominant"),
        (0.4, 0.4, 0.6, "hybrid_dominant"),
        (0.5, 0.51, 0.505, "tie"),
        (0.6, 0.6, 0.3, "near_tie_degraded"),  # both singles tie above hybrid
        (0.6, 0.3, 0.595, "near_tie_safe"),  # hybrid ties the better single
        (0.5, 0.49, 0.48, "near_tie_degraded"),  # chain of sub-eps gaps, but best single - hybrid >= eps
    ],
)
def test_classify_query(b, d, h, expected):
    assert exp1.classify_query(b, d, h, eps=0.02) == expected


def test_degraded_partitions_match_delta_definition():
    """Partition membership must agree exactly with Δ_MaxSingle >= eps."""
    import itertools

    grid = [i / 20 for i in range(21)]
    degraded_parts = {"bm25_dominant", "dense_dominant", "near_tie_degraded"}
    for b, d, h in itertools.product(grid, repeat=3):
        part = exp1.classify_query(b, d, h, 0.02)
        assert (part in degraded_parts) == (max(b, d) - h >= 0.02), (b, d, h, part)


def test_select_alpha_cv_uses_training_folds_only():
    # q0 prefers alpha=1.0 strongly enough to flip the training mean whenever it is a training
    # query; every other query mildly prefers alpha=0.0. So alpha*=1.0 on every fold except q0's.
    idx = [f"q{i}" for i in range(10)]
    df = pd.DataFrame({0.0: [0.0] + [0.6] * 9, 1.0: [1.0] + [0.5] * 9}, index=idx)
    oof, folds = exp1.select_alpha_cv(df, n_folds=5, seed=0)
    assert len(folds) == 5
    for train, test in exp1.create_kfold_splits(idx, 5, 0):
        expected = 1.0 if "q0" in train else 0.0
        assert (oof.loc[test] == expected).all()
    assert oof["q0"] == 0.0  # q0's own preference never leaks into its alpha
