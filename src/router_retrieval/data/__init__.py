"""Data loading, schemas, and dataset splitting utilities."""

from router_retrieval.data.schema import Document, Query, QRel, Candidate
from router_retrieval.data.beir_loader import load_beir_dataset
from router_retrieval.data.splits import create_kfold_splits

__all__ = [
    "Document",
    "Query",
    "QRel",
    "Candidate",
    "load_beir_dataset",
    "create_kfold_splits",
]
