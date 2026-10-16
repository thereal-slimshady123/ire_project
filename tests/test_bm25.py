"""Unit tests for BM25 retriever."""

import math

import pytest

from router_retrieval.data.schema import Document, Query
from router_retrieval.retrieval.bm25 import BM25Retriever, document_text, tokenize

CORPUS = {
    "d1": Document("d1", "vitamin d deficiency and bone health"),
    "d2": Document("d2", "statins lower cholesterol in adults"),
    "d3": Document("d3", "bone density loss in elderly women", title="Osteoporosis"),
    "d4": Document("d4", "the cat sat on the mat"),
}


@pytest.fixture
def bm25():
    r = BM25Retriever(k1=1.2, b=0.75)
    r.index(CORPUS)
    return r


def test_tokenize_stops_and_stems():
    assert tokenize("The Statins are lowering") == ["statin", "lower"]


def test_bm25_indexing_and_search(bm25):
    """Test BM25 index construction and scoring on a synthetic mini corpus."""
    results = bm25.search(Query("q1", "bone loss"), k=10)
    assert [c.doc_id for c in results] == ["d3", "d1"]  # d3 matches both terms
    assert [c.rank for c in results] == [1, 2]
    assert results[0].score > results[1].score > 0

    # Hand-computed score for a single-term query on d2 (Lucene IDF, length-normalized TF).
    n, df, tf = 4, 1, 1
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    doc_lens = [len(tokenize(document_text(d))) for d in CORPUS.values()]
    avgdl = sum(doc_lens) / n
    norm = 1.2 * (1 - 0.75 + 0.75 * doc_lens[1] / avgdl)
    expected = idf * tf * 2.2 / (tf + norm)
    (hit,) = bm25.search(Query("q2", "cholesterol"), k=10)
    assert hit.doc_id == "d2"
    assert hit.score == pytest.approx(expected)


def test_bm25_empty_query(bm25):
    """Test BM25 behavior on empty or out-of-vocabulary query."""
    assert bm25.search(Query("q", ""), k=10) == []
    assert bm25.search(Query("q", "zzzunknown the"), k=10) == []


def test_bm25_save_load_roundtrip(bm25, tmp_path):
    bm25.save(tmp_path / "bm25.pkl")
    loaded = BM25Retriever.load(tmp_path / "bm25.pkl")
    q = Query("q", "bone health")
    assert [(c.doc_id, c.score) for c in loaded.search(q)] == [(c.doc_id, c.score) for c in bm25.search(q)]
