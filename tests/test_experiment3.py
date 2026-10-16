"""Unit tests for Experiment 3's training-fold threshold rule."""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "exp3", Path(__file__).resolve().parents[1] / "scripts" / "run_experiment3_skip_dense.py"
)
exp3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp3)


def test_choose_threshold_most_aggressive_within_tolerance():
    score = np.array([0.9, 0.8, 0.7, 0.6])
    hybrid = np.array([0.5, 0.5, 0.5, 0.5])
    bm25 = np.array([0.5, 0.5, 0.48, 0.1])  # skipping the 4th query costs 0.4/4 = 0.1
    t = exp3.choose_threshold(score, bm25, hybrid, tolerance=-0.01)
    assert t == 0.7  # skip first three (loss -0.005), not the fourth


def test_choose_threshold_never_splits_ties():
    score = np.array([0.9, 0.5, 0.5])
    hybrid = np.zeros(3)
    bm25 = np.array([0.0, 0.0, -1.0])  # skipping both tied queries is too costly
    assert exp3.choose_threshold(score, bm25, hybrid, tolerance=-0.01) == 0.9


def test_choose_threshold_never_skips_when_first_cut_violates():
    assert exp3.choose_threshold(np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0]), -0.01) == float("inf")
