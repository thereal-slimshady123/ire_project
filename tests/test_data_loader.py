"""Unit tests for BEIR data loading and split resolution."""

import json
from pathlib import Path
import pytest
from router_retrieval.data.schema import Document, Query, QRel
from router_retrieval.data.beir_loader import load_corpus, load_queries, load_qrels, load_dataset
from router_retrieval.data.splits import carve_dev_from_train, resolve_splits, load_split_qrels


@pytest.fixture
def mock_beir_dataset(tmp_path: Path):
    """Create a temporary synthetic BEIR dataset directory."""
    dataset_dir = tmp_path / "mock_dataset"
    dataset_dir.mkdir()
    qrels_dir = dataset_dir / "qrels"
    qrels_dir.mkdir()

    # Synthetic corpus
    corpus_records = [
        {"_id": "doc_1", "title": "First Title", "text": "This is the first document text."},
        {"_id": "doc_2", "title": "", "text": "Second document without a title."},
        {"_id": "doc_3", "text": "Third document with missing title key."},
        {"_id": "1004", "title": "Numeric ID", "text": "Document with numeric ID."},
    ]
    with open(dataset_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for r in corpus_records:
            f.write(json.dumps(r) + "\n")

    # Synthetic queries
    query_records = [
        {"_id": "q1", "text": "first query"},
        {"_id": "q2", "text": "second query"},
        {"_id": "q3", "text": "third query"},
        {"_id": "q4", "text": "fourth query"},
        {"_id": "q5", "text": "fifth query"},
        {"_id": "2006", "text": "query with numeric ID"},
    ]
    with open(dataset_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for r in query_records:
            f.write(json.dumps(r) + "\n")

    # Synthetic train qrels (with header)
    with open(qrels_dir / "train.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.write("q1\tdoc_1\t1\n")
        f.write("q2\tdoc_2\t2\n")
        f.write("q3\tdoc_3\t0\n")  # score 0 kept
        f.write("q4\tdoc_1\t1\n")
        f.write("q5\tdoc_2\t1\n")

    # Synthetic test qrels (with header)
    with open(qrels_dir / "test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.write("2006\t1004\t1\n")

    return dataset_dir


def test_load_corpus(mock_beir_dataset: Path):
    corpus = load_corpus(mock_beir_dataset)
    assert len(corpus) == 4
    assert "doc_1" in corpus
    assert corpus["doc_1"].full_text == "First Title This is the first document text."
    assert corpus["doc_2"].title == ""
    assert corpus["doc_2"].full_text == "Second document without a title."
    assert corpus["doc_3"].title == ""
    assert corpus["doc_3"].full_text == "Third document with missing title key."
    assert "1004" in corpus
    assert isinstance(corpus["1004"].doc_id, str)


def test_load_queries(mock_beir_dataset: Path):
    queries = load_queries(mock_beir_dataset)
    assert len(queries) == 6
    assert queries["q1"].text == "first query"
    assert "2006" in queries
    assert isinstance(queries["2006"].query_id, str)


def test_load_qrels(mock_beir_dataset: Path):
    train_qrels = load_qrels(mock_beir_dataset, "train")
    assert len(train_qrels) == 5
    # Confirm score 0 is preserved
    q3_qrel = next(qr for qr in train_qrels if qr.query_id == "q3")
    assert q3_qrel.score == 0
    assert q3_qrel.doc_id == "doc_3"

    test_qrels = load_qrels(mock_beir_dataset, "test")
    assert len(test_qrels) == 1
    assert test_qrels[0].query_id == "2006"
    assert test_qrels[0].doc_id == "1004"
    assert test_qrels[0].score == 1


def test_carve_dev_from_train():
    qids = [f"q{i}" for i in range(10)]
    carved_1 = carve_dev_from_train(qids, dev_fraction=0.2, seed=42)
    carved_2 = carve_dev_from_train(qids, dev_fraction=0.2, seed=42)

    # Determinism
    assert carved_1 == carved_2
    assert len(carved_1["dev"]) == 2
    assert len(carved_1["train"]) == 8

    # Disjointness and sorted
    assert set(carved_1["train"]).isdisjoint(set(carved_1["dev"]))
    assert carved_1["train"] == sorted(carved_1["train"])
    assert carved_1["dev"] == sorted(carved_1["dev"])


def test_resolve_splits_carved(mock_beir_dataset: Path):
    config = {
        "datasets": {
            "mock": {
                "name": "mock",
                "splits": {"train": "train", "dev": None, "test": "test"},
                "dev_from_train_fraction": 0.2,
            }
        }
    }
    splits = resolve_splits("mock", config, seed=42, dataset_dir=mock_beir_dataset)
    assert len(splits["train"]) == 4
    assert len(splits["dev"]) == 1
    assert len(splits["test"]) == 1
    assert set(splits["train"]).isdisjoint(set(splits["dev"]))
    assert splits["test"] == ["2006"]


def test_load_split_qrels_carved(mock_beir_dataset: Path):
    config = {
        "datasets": {
            "mock": {
                "name": "mock",
                "splits": {"train": "train", "dev": None, "test": "test"},
                "dev_from_train_fraction": 0.2,
            }
        }
    }
    splits = resolve_splits("mock", config, seed=42, dataset_dir=mock_beir_dataset)
    dev_qrels = load_split_qrels(mock_beir_dataset, "mock", "dev", config, seed=42)
    train_qrels = load_split_qrels(mock_beir_dataset, "mock", "train", config, seed=42)

    assert len(dev_qrels) == 1
    assert dev_qrels[0].query_id == splits["dev"][0]
    train_qrel_qids = {qr.query_id for qr in train_qrels}
    assert train_qrel_qids == set(splits["train"])


def test_load_dataset(mock_beir_dataset: Path):
    corpus, queries, qrels = load_dataset(mock_beir_dataset, "test")
    assert len(corpus) == 4
    assert len(queries) == 1
    assert "2006" in queries
    assert len(qrels) == 1

