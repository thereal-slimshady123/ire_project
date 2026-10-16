"""Paired significance testing for per-query ranking metrics."""

from typing import Dict, Sequence

import numpy as np
from scipy import stats


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Perform paired bootstrap significance test between two systems on per-query metric.

    The p-value is two-sided and computed under the null by re-centering the bootstrap
    distribution of the mean paired difference at zero (Efron & Tibshirani, 1993, §16.4).

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
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("scores_a and scores_b must be 1-D arrays of equal length")
    n = a.size
    diff = a - b
    observed = float(diff.mean()) if n else 0.0
    if n == 0:
        return {"delta_mean": 0.0, "p_value": 1.0, "ci_95_low": 0.0, "ci_95_high": 0.0, "n": 0}

    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples)
    for start in range(0, n_resamples, 1000):  # chunk to bound memory on large query sets
        stop = min(start + 1000, n_resamples)
        idx = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = diff[idx].mean(axis=1)

    centered = boot - observed
    p_value = (np.sum(np.abs(centered) >= abs(observed)) + 1) / (n_resamples + 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "delta_mean": observed,
        "p_value": float(p_value),
        "ci_95_low": float(lo),
        "ci_95_high": float(hi),
        "n": int(n),
    }


def paired_wilcoxon_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Paired Wilcoxon signed-rank test; zero differences are dropped (zero_method='wilcox').

    Returns:
        Dictionary with 'statistic', 'p_value', and 'n_nonzero' (pairs with a non-zero difference).
    """
    diff = np.asarray(scores_a, dtype=np.float64) - np.asarray(scores_b, dtype=np.float64)
    nonzero = int(np.count_nonzero(diff))
    if nonzero == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_nonzero": 0}
    res = stats.wilcoxon(diff, zero_method="wilcox", alternative=alternative)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n_nonzero": nonzero}
