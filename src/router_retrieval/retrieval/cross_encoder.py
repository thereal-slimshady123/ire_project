"""Cross-encoder reranking backend for escalation."""

from typing import Dict, List, Sequence, Tuple

from router_retrieval.data.schema import Candidate, Document, Query
from router_retrieval.retrieval.bm25 import document_text
from router_retrieval.retrieval.dense import resolve_device


class CrossEncoderReranker:
    """Escalation backend reranking candidate union using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str = "cpu",
        candidate_pool_size: int = 50,
        max_length: int = 512,
    ):
        """
        Initialize CrossEncoderReranker.

        Args:
            model_name: HuggingFace / sentence-transformers cross-encoder model identifier.
            batch_size: Batch size for cross-encoder inference.
            device: Computation device ('cpu', 'cuda' or 'auto').
            candidate_pool_size: Max number of candidates (from the head of the input ranking) to rerank.
            max_length: Max tokens per (query, passage) pair.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.candidate_pool_size = candidate_pool_size
        self.max_length = max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
        return self._model

    def score_pairs(self, pairs: Sequence[Tuple[str, str]], show_progress_bar: bool = False) -> List[float]:
        """Score (query text, passage text) pairs; higher = more relevant."""
        if not pairs:
            return []
        scores = self.model.predict(
            list(pairs), batch_size=self.batch_size, show_progress_bar=show_progress_bar, convert_to_numpy=True
        )
        return [float(s) for s in scores]

    def rerank(
        self,
        query: Query,
        candidates: List[Candidate],
        corpus: Dict[str, Document],
        top_k: int = 10,
    ) -> List[Candidate]:
        """
        Rerank candidate pool using cross-encoder full attention scoring.

        The first `candidate_pool_size` candidates (by input rank) are rescored and reordered;
        the remaining candidates keep their original relative order after them.

        Args:
            query: Query dataclass.
            candidates: Candidate list (e.g. BM25 union Dense), ranked.
            corpus: Document collection to lookup passage texts.
            top_k: Number of reranked candidates to return.

        Returns:
            List of reranked Candidate objects sorted by cross-encoder score descending.
        """
        ranked = sorted(candidates, key=lambda c: c.rank)
        head, tail = ranked[: self.candidate_pool_size], ranked[self.candidate_pool_size :]
        scores = self.score_pairs([(query.text, document_text(corpus[c.doc_id])) for c in head])
        reranked = [c.doc_id for c, _ in sorted(zip(head, scores), key=lambda p: (-p[1], p[0].rank))]
        reranked += [c.doc_id for c in tail]
        score_of = dict(zip((c.doc_id for c in head), scores))
        return [
            Candidate(query_id=query.query_id, doc_id=d, score=score_of.get(d, float("-inf")), rank=r + 1)
            for r, d in enumerate(reranked[:top_k])
        ]
