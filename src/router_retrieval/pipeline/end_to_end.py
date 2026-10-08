"""End-to-end query routing and retrieval execution pipeline."""

from typing import Dict, List, Optional, Tuple
from router_retrieval.data.schema import Candidate, Query, Document
from router_retrieval.retrieval.bm25 import BM25Retriever
from router_retrieval.retrieval.dense import DenseRetriever
from router_retrieval.retrieval.hybrid import HybridRetriever
from router_retrieval.retrieval.cross_encoder import CrossEncoderReranker
from router_retrieval.features.feature_builder import FeatureBuilder
from router_retrieval.router.base import RouterModel
from router_retrieval.router.escalation import EscalationDecider
from router_retrieval.pipeline.latency import LatencyTracker


class EndToEndPipeline:
    """
    Orchestrates full routing pipeline:
    Query -> Feature extraction -> Router decision -> (BM25 | Dense | Hybrid | CrossEncoder Escalation)
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        cross_encoder: CrossEncoderReranker,
        feature_builder: FeatureBuilder,
        router: RouterModel,
        escalation_decider: EscalationDecider,
        corpus: Dict[str, Document],
        latency_tracker: Optional[LatencyTracker] = None,
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.hybrid = hybrid_retriever
        self.cross_encoder = cross_encoder
        self.feature_builder = feature_builder
        self.router = router
        self.escalation_decider = escalation_decider
        self.corpus = corpus
        self.latency_tracker = latency_tracker or LatencyTracker()

    def process_query(
        self,
        query: Query,
        top_k: int = 10,
    ) -> Tuple[List[Candidate], str, bool]:
        """
        Process a single query end-to-end.

        Args:
            query: Input Query.
            top_k: Number of final ranked candidates to return.

        Returns:
            Tuple of:
                - candidates: Final ranked candidate list
                - route_taken: Name of the route chosen ('bm25', 'dense', 'hybrid', or 'escalate')
                - was_escalated: Boolean indicating whether escalation was triggered
        """
        raise NotImplementedError

    def batch_process(
        self,
        queries: List[Query],
        top_k: int = 10,
    ) -> Dict[str, List[Candidate]]:
        """Batch process a collection of queries."""
        raise NotImplementedError
