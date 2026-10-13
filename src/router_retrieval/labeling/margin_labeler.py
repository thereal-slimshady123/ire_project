"""Margin-based route labeler."""

import pandas as pd


class MarginLabeler:
    """
    Labels each query with the best route using argmax over nDCG@10 with margin epsilon.
    If top1 - top2 < epsilon (including exact ties), defaults to 'hybrid' as the safe option.

    Epsilon is selected outside this class: scripts/train_router.py trains a router for each
    epsilon in configs/labels.yaml and keeps the one with the best routed nDCG@10 on dev.
    """

    def __init__(self, epsilon: float = 0.02, default_route: str = "hybrid"):
        """
        Initialize MarginLabeler.

        Args:
            epsilon: Margin threshold below which differences are treated as ties.
            default_route: Default route to assign when within margin epsilon.
        """
        self.epsilon = epsilon
        self.default_route = default_route

    def label_queries(
        self,
        route_scores_df: pd.DataFrame,
        metric: str = "ndcg_cut_10",
    ) -> pd.DataFrame:
        """
        Generate labels for queries given per-route scores.

        Args:
            route_scores_df: DataFrame containing query_id, route, and metric score.
            metric: Target metric to optimize.

        Returns:
            DataFrame with columns [query_id, best_route, margin, top1_score, top2_score],
            where margin = top1_score - top2_score.
        """
        raise NotImplementedError
