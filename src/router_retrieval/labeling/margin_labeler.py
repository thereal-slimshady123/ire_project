"""Margin-based route labeler."""

from typing import Dict, List, Optional
import pandas as pd


class MarginLabeler:
    """
    Labels each query with the best route using argmax over nDCG@10 with margin epsilon.
    If top1 - top2 < epsilon, defaults to 'hybrid' as the safe option.
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
            DataFrame with columns [query_id, best_route, margin, top1_score, top2_score].
        """
        raise NotImplementedError

    def tune_epsilon(
        self,
        train_route_scores_df: pd.DataFrame,
        epsilon_grid: List[float],
        metric: str = "ndcg_cut_10",
    ) -> float:
        """
        Tune epsilon on training data to optimize downstream utility.

        Args:
            train_route_scores_df: Training fold scores.
            epsilon_grid: List of epsilon candidate values.
            metric: Target evaluation metric.

        Returns:
            Optimal epsilon float value.
        """
        raise NotImplementedError
