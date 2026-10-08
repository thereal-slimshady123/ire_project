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
