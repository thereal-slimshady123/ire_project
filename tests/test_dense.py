"""Unit tests for dense retriever."""

import pytest


import numpy as np
import pytest

from router_retrieval.data.schema import Document, Query
from router_retrieval.retrieval.dense import DenseRetriever


class MockSentenceTransformer:
    def __init__(self, dim: int = 4):
        self.dim = dim
        self.max_seq_length = 512

    def encode(self, texts, **kwargs):
        embs = []
        for t in texts:
            # Deterministic vector based on string characters
            seed = sum(ord(c) for c in t) % 1000 + 1
            v = np.sin(np.arange(1, self.dim + 1) * seed).astype(np.float32)
            norm = np.linalg.norm(v)
            embs.append(v / (norm if norm > 0 else 1.0))
        return np.vstack(embs)


@pytest.fixture
def mock_dense():
    retriever = DenseRetriever(model_name="mock-model", device="cpu")
    retriever._model = MockSentenceTransformer(dim=4)
    corpus = {
        "d1": Document("d1", "artificial intelligence in medicine"),
        "d2": Document("d2", "deep learning and neural networks"),
        "d3": Document("d3", "information retrieval systems"),
    }
    retriever.index(corpus, show_progress_bar=False)
    return retriever


def test_dense_indexing_and_search(mock_dense):
    """Test dense retriever encoding and FAISS search on synthetic mini corpus."""
    q = Query("q1", "neural networks in medicine")
    results = mock_dense.search(q, k=3)
    assert len(results) == 3
    assert [c.rank for c in results] == [1, 2, 3]
    assert results[0].score >= results[1].score >= results[2].score
    # Inner product of L2 normalized vectors is in [-1, 1]
    assert all(-1.01 <= c.score <= 1.01 for c in results)


def test_dense_dimension_consistency(mock_dense):
    """Test embedding dimensionality matches index configuration."""
    assert mock_dense._index.d == 4
    assert mock_dense._index.ntotal == 3


def test_dense_save_load_roundtrip(mock_dense, tmp_path):
    """Test serialization and loading of FAISS embeddings index."""
    save_dir = tmp_path / "dense_index"
    mock_dense.save(save_dir)

    loaded = DenseRetriever.load(save_dir, device="cpu")
    loaded._model = mock_dense._model

    q = Query("q1", "neural networks in medicine")
    res_orig = mock_dense.search(q, k=3)
    res_loaded = loaded.search(q, k=3)

    assert [c.doc_id for c in res_orig] == [c.doc_id for c in res_loaded]
    assert [c.score for c in res_orig] == pytest.approx([c.score for c in res_loaded])

