"""Data models, loaders, and split utilities."""

from router_retrieval.data.schema import Document, Query, QRel, Candidate
from router_retrieval.data.beir_loader import load_corpus, load_queries, load_qrels, load_dataset
from router_retrieval.data.splits import carve_dev_from_train, resolve_splits

__all__ = [
    "Document",
    "Query",
    "QRel",
    "Candidate",
    "load_corpus",
    "load_queries",
    "load_qrels",
    "load_dataset",
    "carve_dev_from_train",
    "resolve_splits",
]
