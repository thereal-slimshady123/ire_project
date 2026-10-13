"""Cross-encoder reranking backend for escalation."""

from typing import List, Dict
from router_retrieval.data.schema import Candidate, Query, Document


class CrossEncoderReranker:
    """
    Escalation backend.

    Input pool = union of BM25 top-100 and dense top-100, ordered by hybrid (alpha*) score.
    The top `rerank_depth` documents are reranked by the cross-encoder; the remaining
    hybrid-ordered documents are appended after them, so `output_depth` results are returned
    (MAP@100 and Recall@100 need a full-depth list).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str = "cuda",
        rerank_depth: int = 50,
        output_depth: int = 100,
    ):
        """
        Initialize CrossEncoderReranker.

        Args:
            model_name: HuggingFace / sentence-transformers cross-encoder model identifier.
            batch_size: Batch size for cross-encoder inference.
            device: Computation device ('cuda' or 'cpu'; falls back to cpu without CUDA).
            rerank_depth: Number of top hybrid-ordered candidates scored by the cross-encoder.
            output_depth: Length of the returned list.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.rerank_depth = rerank_depth
        self.output_depth = output_depth

    def rerank(
        self,
        query: Query,
        hybrid_candidates: List[Candidate],
        corpus: Dict[str, Document],
    ) -> List[Candidate]:
        """
        Rerank the head of the hybrid-ordered union pool with the cross-encoder.

        Args:
            query: Query dataclass.
            hybrid_candidates: Hybrid-fused union pool, sorted by hybrid score descending.
            corpus: Document collection to look up passage texts (Document.full_text).

        Returns:
            Up to `output_depth` Candidates: the reranked top `rerank_depth` followed by the rest
            in hybrid order. Scores are strictly decreasing with rank (cross-encoder and hybrid
            scores are on different scales, and pytrec_eval sorts by score, not by rank).
        """
        raise NotImplementedError
