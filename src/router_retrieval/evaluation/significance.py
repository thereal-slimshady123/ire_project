"""Paired bootstrap significance testing for ranking metrics."""

from typing import Dict, Sequence, Tuple
import numpy as np


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Perform paired bootstrap significance test between two systems on per-query metric.

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
