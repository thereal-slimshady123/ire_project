"""Per-query, per-route retrieval evaluation."""

from typing import Dict, List
import pandas as pd
from router_retrieval.data.schema import Candidate, QRel


def evaluate_per_query_per_route(
    route_runs: Dict[str, Dict[str, List[Candidate]]],
    qrels: List[QRel],
    metrics: List[str] = None,
) -> pd.DataFrame:
    """
    Evaluate ranking performance per query across all cheap routes (BM25, Dense, Hybrid).

    Args:
        route_runs: Dictionary mapping route_name -> {query_id -> List[Candidate]}.
        qrels: Ground truth relevance judgments.
        metrics: List of metrics (e.g. ['ndcg_cut_10', 'recip_rank', 'map', 'recall_100']).

    Returns:
        DataFrame with columns [query_id, route, ndcg_cut_10, recip_rank, map, recall_100].
    """
    raise NotImplementedError
