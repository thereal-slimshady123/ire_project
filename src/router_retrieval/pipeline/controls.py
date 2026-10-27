"""Baseline and control pipelines for rigorous comparison."""

from typing import Dict, List, Optional
import hashlib
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
        b = self.bm25.search(query, k=max(100, self.cross_encoder.candidate_pool_size))
        d = self.dense.search(query, k=max(100, self.cross_encoder.candidate_pool_size))
        candidates = HybridRetriever(self.bm25, self.dense, alpha=0.5).combine_candidates(
            b, d, k=max(100, self.cross_encoder.candidate_pool_size))
        return self.cross_encoder.rerank(query, candidates, self.corpus, top_k)


def _selected(query_id: str, seed: int, rate: float) -> bool:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    digest = hashlib.sha256(f"{seed}:{query_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


class RandomEscalationPipeline:
    """Deterministically samples queries for escalation at an approximate target rate."""

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
        candidates = self.hybrid.search(query, k=max(100, self.cross_encoder.candidate_pool_size))
        if _selected(query.query_id, self.seed, self.escalation_rate):
            return self.cross_encoder.rerank(query, candidates, self.corpus, top_k)
        return candidates[:top_k]


class AlwaysHybridRandomRerankPipeline:
    """Always-hybrid baseline with seeded random reranking at an approximate target rate."""

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
        candidates = self.hybrid.search(query, k=max(100, self.cross_encoder.candidate_pool_size))
        if _selected(query.query_id, self.seed, self.rerank_rate):
            return self.cross_encoder.rerank(query, candidates, self.corpus, top_k)
        return candidates[:top_k]
