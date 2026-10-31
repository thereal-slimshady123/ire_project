"""Margin-based route labeler."""

from typing import Callable, List, Optional

import numpy as np
import pandas as pd


class MarginLabeler:
    """
    Labels each query with the best route using argmax over nDCG@10 with margin epsilon.
    If top1 - top2 < epsilon, defaults to 'hybrid' as the safe option.

    With `margin_to="default"` the margin is instead top1 - score(default route): a query is labelled with
    a non-default route only if that route beats the default by >= epsilon, so two non-default routes that
    are close to each other but both far above the default are no longer labelled with the default.
    """

    def __init__(self, epsilon: float = 0.02, default_route: str = "hybrid", margin_to: str = "runner_up"):
        """
        Initialize MarginLabeler.

        Args:
            epsilon: Margin threshold below which differences are treated as ties.
            default_route: Default route to assign when within margin epsilon.
            margin_to: "runner_up" (top1 - top2; the pre-registered Experiment 2 rule) or "default"
                (top1 - score of `default_route`).
        """
        if margin_to not in ("runner_up", "default"):
            raise ValueError(f"margin_to must be 'runner_up' or 'default', got {margin_to!r}")
        self.epsilon = epsilon
        self.default_route = default_route
        self.margin_to = margin_to

    def label_wide(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Label from a wide frame (index = query_id, one column per route holding the metric).

        Exact ties for first place go to `default_route` when it is among the tied routes.
        """
        routes = list(scores.columns)
        values = scores.to_numpy(dtype=np.float64)
        order = np.argsort(-values, axis=1, kind="stable")
        top1 = values[np.arange(len(values)), order[:, 0]]
        top2 = values[np.arange(len(values)), order[:, 1]] if len(routes) > 1 else top1
        margin = top1 - top2
        best = np.array(routes, dtype=object)[order[:, 0]]
        default_is_top = (
            values[:, routes.index(self.default_route)] >= top1 if self.default_route in routes else False
        )
        if self.margin_to == "default" and self.default_route in routes:
            margin = top1 - values[:, routes.index(self.default_route)]
        best = np.where((margin < self.epsilon) | default_is_top, self.default_route, best)
        return pd.DataFrame(
            {"best_route": best, "margin": margin, "top1_score": top1, "top2_score": top2}, index=scores.index
        )

    def label_queries(
        self,
        route_scores_df: pd.DataFrame,
        metric: str = "ndcg_cut_10",
    ) -> pd.DataFrame:
        """
        Generate labels for queries given per-route scores.

        Args:
            route_scores_df: DataFrame containing query_id, route, and metric score (long format).
            metric: Target metric to optimize.

        Returns:
            DataFrame with columns [query_id, best_route, margin, top1_score, top2_score].
        """
        wide = route_scores_df.pivot(index="query_id", columns="route", values=metric)
        return self.label_wide(wide).reset_index()

    def tune_epsilon(
        self,
        train_route_scores_df: pd.DataFrame,
        epsilon_grid: List[float],
        metric: str = "ndcg_cut_10",
        utility_fn: Optional[Callable[[float], float]] = None,
    ) -> float:
        """
        Tune epsilon on training data to optimize downstream utility.

        `utility_fn(epsilon) -> float` should train/evaluate a router on training data only
        (e.g. inner cross-validation) and return the routed metric. Without it, there is no
        label-only notion of utility, so the configured epsilon is kept.

        Args:
            train_route_scores_df: Training fold scores.
            epsilon_grid: List of epsilon candidate values.
            metric: Target evaluation metric.
            utility_fn: Callable scoring a candidate epsilon on training data.

        Returns:
            Optimal epsilon float value (ties -> the smaller epsilon).
        """
        if utility_fn is None:
            return self.epsilon
        scores = [(utility_fn(e), -e) for e in epsilon_grid]
        best = max(scores)
        self.epsilon = -best[1]
        return self.epsilon
