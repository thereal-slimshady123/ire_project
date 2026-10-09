"""Data schema definitions for router_retrieval."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Document:
    """Document representation in corpus."""
    doc_id: str
    text: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """Query representation."""
    query_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QRel:
    """Ground truth relevance judgment."""
    query_id: str
    doc_id: str
    score: int = 1


@dataclass
class Candidate:
    """Retrieved candidate document with score and rank."""
    query_id: str
    doc_id: str
    score: float
    rank: int = 1

