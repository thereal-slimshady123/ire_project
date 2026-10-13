"""Regret computation for regret-weighted loss."""

import pandas as pd


def compute_regret_weights(
    route_scores_df: pd.DataFrame,
    metric: str = "ndcg_cut_10",
) -> pd.DataFrame:
    """
    Compute per-query regret for each route.
    regret(query, route) = max_{r} nDCG(query, r) - nDCG(query, route)
    Used only to build training sample weights (configs/router.yaml sample_weighting).

    Args:
        route_scores_df: DataFrame with columns [query_id, route, <metric>].
        metric: Column name representing evaluation metric score.

    Returns:
        DataFrame containing query_id and regret columns:
        [query_id, regret_bm25, regret_dense, regret_hybrid, max_regret].
    """
    raise NotImplementedError


def compute_sample_weights(
    labels_df: pd.DataFrame,
    class_balance: bool = True,
    use_regret_weights: bool = True,
    regret_weight_floor: float = 0.05,
) -> pd.Series:
    """
    Build per-query training weights (configs/router.yaml sample_weighting):
        weight = class_balance_weight(label) * (margin + regret_weight_floor), rescaled to mean 1.
    This is the ONLY place class balancing happens; routers must not add their own class_weight.

    Args:
        labels_df: Labels with columns [query_id, best_route, margin].
        class_balance: Multiply by n_samples / (n_classes * n_samples_in_class).
        use_regret_weights: Multiply by (margin + regret_weight_floor).
        regret_weight_floor: Keeps near-tie queries from getting zero weight.

    Returns:
        Series of weights indexed like labels_df.
    """
    raise NotImplementedError
