"""Shared data schemas and dataclasses across retrieval, labeling, and features."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Document:
    """Represents a corpus document."""
    doc_id: str
    text: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """Represents an evaluation or training query."""
    query_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QRel:
    """Represents ground truth relevance judgment."""
    query_id: str
    doc_id: str
    relevance: int


@dataclass
class Candidate:
    """Represents a retrieved or reranked candidate document for a query."""
    query_id: str
    doc_id: str
    score: float
    rank: int
