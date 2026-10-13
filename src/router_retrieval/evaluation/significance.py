"""Paired bootstrap significance testing for ranking metrics."""

from typing import Dict, List, Sequence
import numpy as np


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Perform two-sided paired bootstrap significance test between two systems on a per-query metric.
    Both sequences must be aligned on the same query order.
    Not meaningful against the oracle router: oracle >= hybrid on every query by construction.

    Args:
        scores_a: System A per-query metric scores (e.g., router nDCG@10).
        scores_b: System B per-query metric scores (e.g., hybrid baseline nDCG@10).
        n_resamples: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - 'delta_mean': float (mean(A) - mean(B))
            - 'p_value': float
            - 'ci_95_low': float
            - 'ci_95_high': float
    """
    raise NotImplementedError


def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """
    Holm-Bonferroni correction across all comparisons made on one dataset.

    Args:
        p_values: comparison name -> raw p-value.
        alpha: Family-wise error rate.

    Returns:
        comparison name -> {'p_value': raw, 'p_adjusted': adjusted, 'significant': 0.0 or 1.0}.
    """
    raise NotImplementedError
