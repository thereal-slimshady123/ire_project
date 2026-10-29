"""Tests for FeatureBuilder: parity with the Experiment 1 feature path, bug regressions, Contract 2 I/O."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router_retrieval.data.schema import Candidate, Query
from router_retrieval.features.feature_builder import FEATURE_COLUMNS, FeatureBuilder
from router_retrieval.retrieval.bm25 import BM25Retriever
from tests.track2_fixtures import toy_corpus


def _cands(qid, ids, scores):
    return [Candidate(qid, d, s, i + 1) for i, (d, s) in enumerate(zip(ids, scores))]


@pytest.fixture(scope="module")
def setup():
    corpus = toy_corpus()
    bm25 = BM25Retriever()
    bm25.index(corpus)
    queries = {"a": Query("a", "Kubernetes error 137 memory"), "b": Query("b", "Metformin glucose in diabetes")}
    bm25_run = bm25.batch_search(list(queries.values()), 5)
    dense_run = {"a": _cands("a", ["d4", "d6", "d5"], [0.9, 0.8, 0.7]), "b": _cands("b", ["d1", "d2", "d3"], [0.8, 0.79, 0.5])}
    return corpus, bm25, queries, bm25_run, dense_run


def test_columns_and_order(setup):
    corpus, bm25, queries, bm25_run, dense_run = setup
    fb = FeatureBuilder.from_bm25(bm25, corpus)
    df = fb.build_feature_matrix(list(queries.values()), bm25_run, dense_run)
    assert list(df.columns) == FEATURE_COLUMNS
    assert list(df.index) == ["a", "b"] and df.index.name == "query_id"
    assert np.isfinite(df.to_numpy()).all()


def test_rare_term_density_is_computed(setup):
    """Regression: the builder used to drop rare_threshold, leaving rare_term_density identically 0."""
    corpus, bm25, queries, bm25_run, dense_run = setup
    from router_retrieval.features.query_features import rare_idf_threshold

    fb = FeatureBuilder.from_bm25(bm25, corpus)
    assert fb.rare_threshold == pytest.approx(rare_idf_threshold(bm25.num_docs))
    fb.rare_threshold = 0.5  # every in-vocabulary term of this tiny corpus is "rare" at this threshold
    assert fb.extract_features(queries["a"], bm25_run["a"], dense_run["a"])["rare_term_density"] > 0.0
    fb.rare_threshold = None  # no threshold supplied -> documented 0
    assert fb.extract_features(queries["a"], bm25_run["a"], dense_run["a"])["rare_term_density"] == 0.0


def test_matches_experiment1_feature_path(setup):
    """The library path must reproduce scripts/run_experiment1_hybrid_degradation.compute_features exactly."""
    spec = importlib.util.spec_from_file_location(
        "exp1", Path(__file__).resolve().parents[1] / "scripts" / "run_experiment1_hybrid_degradation.py")
    exp1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp1)
    corpus, bm25, queries, bm25_run, dense_run = setup
    ref = exp1.compute_features(queries, corpus, bm25, bm25_run, dense_run)
    ours = FeatureBuilder.from_bm25(bm25, corpus).build_feature_matrix(list(queries.values()), bm25_run, dense_run)
    pd.testing.assert_frame_equal(ours.sort_index(), ref.sort_index()[FEATURE_COLUMNS], check_dtype=False, atol=1e-9)


def test_empty_candidate_lists_do_not_crash(setup):
    corpus, bm25, queries, _, _ = setup
    f = FeatureBuilder.from_bm25(bm25, corpus).extract_features(queries["a"], [], [])
    assert set(f) == set(FEATURE_COLUMNS) and f["bm25_top10_query_term_coverage"] == 0.0


def test_contract2_roundtrip(tmp_path, setup):
    corpus, bm25, queries, bm25_run, dense_run = setup
    fb = FeatureBuilder.from_bm25(bm25, corpus, cache_dir=tmp_path)
    df = fb.build_feature_matrix(list(queries.values()), bm25_run, dense_run)
    fb.save_features(df, tmp_path / "toy.parquet")
    raw = pd.read_parquet(tmp_path / "toy.parquet")
    assert raw["query_id"].dtype == object and raw["query_id"].tolist() == ["a", "b"]
    assert raw["max_idf"].dtype == np.float32
    assert (raw["jaccard_overlap"] == raw["jaccard_overlap_top10"]).all()  # Contract 2 alias
    assert (raw["rank_correlation"] == raw["rank_correlation_spearman"]).all()
    loaded = fb.load_features(tmp_path / "toy.parquet")
    assert list(loaded.index) == ["a", "b"]
    np.testing.assert_allclose(loaded[FEATURE_COLUMNS].to_numpy(), df.to_numpy(), rtol=1e-5, atol=1e-6)


def test_cached_matrix_builds_once(tmp_path, setup):
    calls = []
    fb = FeatureBuilder(cache_dir=tmp_path)
    frame = pd.DataFrame({"query_length": [3.0]}, index=pd.Index(["x"], name="query_id"))
    build = lambda: calls.append(1) or frame
    fb.cached_matrix("ds", build)
    again = fb.cached_matrix("ds", build)
    assert len(calls) == 1 and list(again.index) == ["x"]
