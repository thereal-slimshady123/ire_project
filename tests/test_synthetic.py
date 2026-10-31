"""Tests for synthetic query generation, leakage control and labelling (offline, no ML models)."""

import numpy as np
import pandas as pd
import pytest

from router_retrieval.data.schema import QRel, Query
from router_retrieval.features.feature_builder import FEATURE_COLUMNS
from router_retrieval.retrieval.bm25 import BM25Retriever
from router_retrieval.synthetic import (
    ExtractiveLexicalGenerator, MixedGenerator, SyntheticQuery, allocate_budget, assert_no_leakage,
    build_route_table, clean_queries, eligible_doc_ids, excluded_doc_ids, load_synthetic, plan_queries,
    save_synthetic, style_capacity, subsample_synthetic,
)
from router_retrieval.synthetic.generator import content_overlap, generate_low_overlap
from router_retrieval.synthetic.sampler import _norm
from tests.track2_fixtures import ALPHAS, toy_corpus


# -------------------------------------------------------------------- lexical generator

def test_extractive_generator_prefers_rare_tokens_and_keeps_identifiers():
    idf = {"the": 0.1, "error": 3.0, "memori": 4.0, "kubernet": 8.0, "contain": 2.0}
    gen = ExtractiveLexicalGenerator(idf.get, min_terms=3, max_terms=3, oov_idf=5.0)
    [q] = gen.generate(["Kubernetes error 137 means the container exceeded memory"], "lexical")
    toks = q.split()
    assert len(toks) == 3
    assert "137" in toks and "Kubernetes" in toks           # OOV identifier + highest-IDF word
    assert q.split() == sorted(q.split(), key=lambda t: "Kubernetes error 137 means the container exceeded memory".split().index(t))  # passage order
    with pytest.raises(ValueError):
        gen.generate(["x"], "semantic")


def test_extractive_generator_is_deterministic_and_handles_empty_passages():
    mk = lambda: ExtractiveLexicalGenerator({"alpha": 1.0}.get, seed=3)
    texts = ["alpha beta gamma delta epsilon zeta", "the of and"]
    assert mk().generate(texts, "lexical") == mk().generate(texts, "lexical")
    assert mk().generate(["the of and"], "lexical") == [""]


def test_extractive_generator_pool_factor_samples_beyond_the_rarest_terms():
    idf = {w: float(i) for i, w in enumerate("a b c d e f g h i j".split(), start=1)}
    text = "a b c d e f g h i j"
    rarest = ExtractiveLexicalGenerator(idf.get, min_terms=3, max_terms=3).generate([text] * 20, "lexical")
    assert set(rarest) == {"h i j"}
    harder = ExtractiveLexicalGenerator(idf.get, min_terms=3, max_terms=3, pool_factor=3, seed=0).generate([text] * 20, "lexical")
    assert len(set(harder)) > 1 and all(set(q.split()) <= set("b c d e f g h i j".split()) for q in harder)
    with pytest.raises(ValueError):
        ExtractiveLexicalGenerator(idf.get, pool_factor=0.5)


def test_generate_low_overlap_resamples_and_drops():
    class Scripted:
        def __init__(self):
            self.calls = 0

        def generate(self, passages, style):
            self.calls += 1
            # passage 0: overlapping on the first try, clean on the second; passage 1 always overlaps
            return ["kubernetes memory" if (p.startswith("Kube") and self.calls == 1) else
                    ("why do programs crash" if p.startswith("Kube") else "glucose metformin") for p in passages]

    gen = Scripted()
    passages = ["Kubernetes memory limits", "Metformin lowers glucose"]
    assert content_overlap("kubernetes memory", passages[0]) == 2
    out = generate_low_overlap(gen, passages, "semantic", max_overlap=0, attempts=3)
    assert out == ["why do programs crash", ""]
    assert gen.calls == 3


def test_mixed_generator_dispatches_by_style():
    class Fixed:
        def __init__(self, tag): self.tag = tag
        def generate(self, passages, style): return [f"{self.tag}:{p}" for p in passages]
    g = MixedGenerator(Fixed("lex"), Fixed("sem"))
    assert g.generate(["p"], "lexical") == ["lex:p"] and g.generate(["p"], "semantic") == ["sem:p"]


# ------------------------------------------------------------------------ sampler

def test_exclusion_and_eligibility():
    corpus = toy_corpus()
    qrels = [QRel("q1", "d1", 2), QRel("q1", "d2", 0), QRel("q2", "d3", 1)]
    excluded = excluded_doc_ids(qrels)
    assert excluded == {"d1", "d3"}                          # relevance-0 judgments do not exclude
    eligible = eligible_doc_ids(corpus, excluded, min_words=8)
    assert "d1" not in eligible and "d3" not in eligible
    assert "d11" not in eligible                             # too short
    assert eligible == sorted(eligible) and "d2" in eligible


def test_plan_is_balanced_deterministic_and_bounded():
    eligible = [f"d{i}" for i in range(50)]
    plan = plan_queries(eligible, 60, 0.5, seed=1)
    assert plan == plan_queries(eligible, 60, 0.5, seed=1)
    styles = [s for _, s in plan]
    assert styles.count("lexical") == 30 and styles.count("semantic") == 30
    per_style = {s: [d for d, st in plan if st == s] for s in ("lexical", "semantic")}
    assert all(len(set(v)) == len(v) for v in per_style.values())     # no repeats within a style
    assert style_capacity(50, 0.5) == 100 and style_capacity(50, 1.0) == 50
    with pytest.raises(ValueError):
        plan_queries(eligible, 101, 0.5)
    assert {s for _, s in plan_queries(eligible, 10, 1.0)} == {"lexical"}


def test_allocate_budget_caps_and_redistributes():
    assert allocate_budget({"a": 100, "b": 100, "c": 100}, 90) == {"a": 30, "b": 30, "c": 30}
    alloc = allocate_budget({"a": 10, "b": 100, "c": 100}, 90)
    assert alloc["a"] == 10 and alloc["b"] + alloc["c"] == 80 and sum(alloc.values()) == 90
    assert allocate_budget({"a": 5}, 100) == {"a": 5}
    assert allocate_budget({"a": 0, "b": 4}, 3) == {"a": 0, "b": 3}


def test_clean_queries_drops_short_duplicate_and_natural_copies():
    keep = clean_queries(["a", "Metformin  glucose", "metformin glucose", "aspirin platelet", "something new here"],
                         natural_texts=["Aspirin platelet"])
    assert keep == [False, True, False, False, True]


def test_leakage_guard():
    assert_no_leakage(["d2", "d5"], {"d1", "d3"})
    with pytest.raises(ValueError, match="evaluation-relevant"):
        assert_no_leakage(["d2", "d1"], {"d1", "d3"})


def test_pipeline_never_samples_excluded_passages():
    corpus = toy_corpus()
    excluded = excluded_doc_ids([QRel("q", d, 1) for d in ("d1", "d2", "d3", "d4")])
    eligible = eligible_doc_ids(corpus, excluded, min_words=8)
    for seed in range(5):
        plan = plan_queries(eligible, 8, 0.5, seed)
        assert_no_leakage([d for d, _ in plan], excluded)


# --------------------------------------------------------------- labelling & persistence

class _ShuffledDense:
    """Stand-in dense retriever: BM25 with different parameters, so the two routes differ."""
    def __init__(self, corpus):
        self.r = BM25Retriever(k1=2.0, b=0.2)
        self.r.index(corpus)

    def batch_search(self, queries, k=100):
        return self.r.batch_search(queries, k)


@pytest.fixture(scope="module")
def table():
    corpus = toy_corpus()
    bm25 = BM25Retriever()
    bm25.index(corpus)
    qs = [SyntheticQuery("syn-0", "kubernetes OOMKilled memory", "d4", "lexical", "toy"),
          SyntheticQuery("syn-1", "why does my container die when it runs out of memory", "d4", "semantic", "toy"),
          SyntheticQuery("syn-2", "metformin AMPK glucose", "d2", "lexical", "toy"),
          SyntheticQuery("syn-3", "zzzz qqqq", "d5", "semantic", "toy")]       # nothing matches
    return build_route_table(qs, bm25, _ShuffledDense(corpus), ALPHAS, corpus, k=5)


def test_route_table_schema_matches_natural_tables(table):
    expected = ["dataset", "ndcg_bm25", "ndcg_dense", *[f"ndcg_hyb_{a}" for a in ALPHAS], *FEATURE_COLUMNS,
                "style", "source_doc", "query_text"]
    assert list(table.columns) == expected
    assert list(table.index) == ["syn-0", "syn-1", "syn-2", "syn-3"] and table.index.name == "query_id"
    assert table[["ndcg_bm25", "ndcg_dense"]].to_numpy().min() >= 0 and table[["ndcg_bm25", "ndcg_dense"]].to_numpy().max() <= 1


def test_labels_score_the_source_passage_only(table):
    # exact lexical match -> source passage retrieved at rank 1 -> nDCG@10 == 1 for BM25 and alpha=0 hybrid
    assert table.loc["syn-0", "ndcg_bm25"] == pytest.approx(1.0)
    assert table.loc["syn-0", "ndcg_hyb_0.0"] == pytest.approx(1.0)      # alpha=0 reproduces BM25
    assert table.loc["syn-2", "ndcg_bm25"] == pytest.approx(1.0)
    assert (table.loc["syn-3", [c for c in table.columns if c.startswith("ndcg_")]] == 0).all()
    assert np.isfinite(table[FEATURE_COLUMNS].to_numpy()).all()


def test_save_load_and_nested_balanced_subsampling(tmp_path, table):
    save_synthetic(table, tmp_path / "toy.parquet")
    back = load_synthetic(tmp_path / "toy.parquet")
    assert list(back.index) == list(table.index) and back["style"].tolist() == table["style"].tolist()
    big = pd.concat([table.assign(_i=i).rename(index=lambda q, i=i: f"{q}-{i}") for i in range(10)])
    s2, s4, s8 = (subsample_synthetic(big, n, seed=1) for n in (2, 4, 8))
    assert (s2["style"].value_counts() == 1).all() and (s8["style"].value_counts() == 4).all()
    assert set(s2.index) <= set(s4.index) <= set(s8.index)                 # nested
    assert len(subsample_synthetic(big, 10_000, seed=1)) == len(big) and subsample_synthetic(big, 0).empty
