"""Baseline and control escalation policies for rigorous comparison.

All controls operate on whole query sets, because "same escalation rate" means escalating
exactly the same NUMBER of queries as the router (not a per-query coin flip). Each policy
returns query_id -> decision, where a decision is 'bm25', 'dense', 'hybrid' or 'escalate'.
Final runs are assembled from precomputed per-route runs and the cross-encoder run.
"""

from typing import Dict, List, Sequence
from router_retrieval.data.schema import Candidate


class AlwaysRerankPipeline:
    """Every query is escalated to the cross-encoder (cost and quality upper bound)."""

    def decide(self, query_ids: Sequence[str]) -> Dict[str, str]:
        """Return 'escalate' for every query."""
        raise NotImplementedError


class RandomEscalationPipeline:
    """
    Keeps the router's route for every query, but replaces WHICH queries are escalated:
    exactly n_escalate queries, chosen uniformly at random. Tests whether the router's
    confidence picks better queries to escalate than chance.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def decide(self, router_routes: Dict[str, str], n_escalate: int) -> Dict[str, str]:
        """
        Args:
            router_routes: query_id -> router's argmax cheap route (before escalation).
            n_escalate: Number of queries the router escalated at the same tau.

        Returns:
            query_id -> decision.
        """
        raise NotImplementedError


class AlwaysHybridRandomRerankPipeline:
    """
    No routing at all: every query uses hybrid, and exactly n_escalate random queries are
    reranked. Tests whether routing adds anything beyond spending the same reranking budget.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def decide(self, query_ids: Sequence[str], n_escalate: int) -> Dict[str, str]:
        """
        Args:
            query_ids: All evaluated query IDs.
            n_escalate: Number of queries the router escalated at the same tau.

        Returns:
            query_id -> decision ('hybrid' or 'escalate').
        """
        raise NotImplementedError


def assemble_run(
    decisions: Dict[str, str],
    route_runs: Dict[str, Dict[str, List[Candidate]]],
    rerank_run: Dict[str, List[Candidate]],
) -> Dict[str, List[Candidate]]:
    """
    Build the final run: for each query take the list from route_runs[decision], or from
    rerank_run if the decision is 'escalate'. Lists are already at retrieval depth 100.

    Args:
        decisions: query_id -> decision.
        route_runs: route name -> query_id -> candidates ('bm25', 'dense', 'hybrid').
        rerank_run: query_id -> cross-encoder reranked candidates.

    Returns:
        query_id -> candidates.
    """
    raise NotImplementedError
