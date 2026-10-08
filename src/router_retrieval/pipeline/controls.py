"""Baseline and control pipelines for rigorous comparison."""

from typing import Dict, List, Optional
from router_retrieval.data.schema import Candidate, Query, Document
from router_retrieval.retrieval.bm25 import BM25Retriever
from router_retrieval.retrieval.dense import DenseRetriever
from router_retrieval.retrieval.hybrid import HybridRetriever
from router_retrieval.retrieval.cross_encoder import CrossEncoderReranker


class AlwaysRerankPipeline:
    """Every query is reranked through the cross-encoder backend."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        cross_encoder: CrossEncoderReranker,
        corpus: Dict[str, Document],
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.cross_encoder = cross_encoder
        self.corpus = corpus

    def process_query(self, query: Query, top_k: int = 10) -> List[Candidate]:
        """Process query by always routing through cross-encoder."""
        raise NotImplementedError


class RandomEscalationPipeline:
    """Escalates a random subset of queries at the exact same rate as the router."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        cross_encoder: CrossEncoderReranker,
        corpus: Dict[str, Document],
        escalation_rate: float = 0.10,
        seed: int = 42,
    ):
        self.hybrid = hybrid_retriever
        self.cross_encoder = cross_encoder
        self.corpus = corpus
        self.escalation_rate = escalation_rate
        self.seed = seed

    def process_query(self, query: Query, top_k: int = 10) -> List[Candidate]:
        """Process query with random escalation."""
        raise NotImplementedError


class AlwaysHybridRandomRerankPipeline:
    """Always-hybrid baseline with random cross-encoder reranking on a matching fraction."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        cross_encoder: CrossEncoderReranker,
        corpus: Dict[str, Document],
        rerank_rate: float = 0.10,
        seed: int = 42,
    ):
        self.hybrid = hybrid_retriever
        self.cross_encoder = cross_encoder
        self.corpus = corpus
        self.rerank_rate = rerank_rate
        self.seed = seed

    def process_query(self, query: Query, top_k: int = 10) -> List[Candidate]:
        """Process query with random subset reranked."""
        raise NotImplementedError
