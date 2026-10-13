"""Oracle router that selects the ground-truth optimal route per query."""

from typing import Dict, List, Sequence
from router_retrieval.data.schema import Candidate, Query


class OracleRouter:
    """
    Oracle router baseline.
    Upper-bound control that looks up the true best route per query without escalation.
    """

    def __init__(self, ground_truth_labels: Dict[str, str]):
        """
        Initialize Oracle Router.

        Args:
            ground_truth_labels: Map from query_id to optimal route name.
        """
        self.ground_truth_labels = ground_truth_labels

    def route_and_retrieve(
        self,
        query: Query,
        candidate_pools: Dict[str, List[Candidate]],
    ) -> List[Candidate]:
        """
        Select candidates corresponding to ground-truth best route for the query.

        Args:
            query: Query object.
            candidate_pools: Dict mapping route name to candidate list.

        Returns:
            Candidates from the optimal route.
        """
        raise NotImplementedError


def oracle_escalation(
    routed_scores: Dict[str, float],
    rerank_scores: Dict[str, float],
    escalation_rate: float,
) -> List[str]:
    """
    Escalation ceiling (evaluation only, no model): with the router's routing held fixed,
    escalate the escalation_rate fraction of queries whose true cross-encoder gain
    (rerank_score - routed_score) is largest. Gives the best quality any escalation rule
    could reach at that budget, for comparison with confidence-based escalation (H3).

    Args:
        routed_scores: query_id -> nDCG@10 of the route the router chose.
        rerank_scores: query_id -> nDCG@10 of the cross-encoder (always-rerank) run.
        escalation_rate: Fraction of queries to escalate (count = round(rate * n_queries)).

    Returns:
        Query IDs to escalate.
    """
    raise NotImplementedError
