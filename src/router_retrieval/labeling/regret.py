"""Regret computation for regret-weighted loss."""

import pandas as pd


def compute_regret_weights(
    route_scores_df: pd.DataFrame,
    metric: str = "ndcg_cut_10",
) -> pd.DataFrame:
    """
    Compute per-query regret for each route.
    regret(query, route) = max_{r} nDCG(query, r) - nDCG(query, route)

    Args:
        route_scores_df: DataFrame with columns [query_id, route, score].
        metric: Column name representing evaluation metric score.

    Returns:
        DataFrame containing query_id and regret columns:
        [query_id, regret_bm25, regret_dense, regret_hybrid, max_regret].
    """
    raise NotImplementedError
