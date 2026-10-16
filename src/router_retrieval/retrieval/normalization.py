"""Score normalization utilities for sparse and dense retrieval scores."""

from dataclasses import replace
from typing import List, Optional

import numpy as np

from router_retrieval.data.schema import Candidate

# Smoothing constant delta in the min-max denominator (guards single-candidate / all-equal lists).
MIN_MAX_DELTA = 1e-9


def min_max_normalize(candidates: List[Candidate], floor: Optional[float] = None) -> List[Candidate]:
    """
    Min-max normalize candidate scores to [0, 1] range.

    s~(d) = (s(d) - lo) / (max s - lo + delta), with lo = min s over this query's candidate list,
    or `floor` when given. Pass the score an *unretrieved* document is known to have as `floor`
    when that is lower than the list minimum (e.g. 0.0 for a BM25 list truncated because no other
    document matched): otherwise a single-candidate list normalizes to 0 and the lowest retrieved
    document becomes indistinguishable from documents that were never retrieved.

    Args:
        candidates: List of Candidate objects for a query.
        floor: Optional lower anchor (must be <= every score in the list).

    Returns:
        List of Candidate objects with normalized scores.
    """
    if not candidates:
        return []
    scores = np.array([c.score for c in candidates], dtype=np.float64)
    lo = scores.min() if floor is None else min(floor, scores.min())
    hi = scores.max()
    norm = (scores - lo) / (hi - lo + MIN_MAX_DELTA)
    return [replace(c, score=float(s)) for c, s in zip(candidates, norm)]


def z_score_normalize(candidates: List[Candidate]) -> List[Candidate]:
    """
    Z-score normalize candidate scores (zero mean, unit variance).

    Args:
        candidates: List of Candidate objects for a query.

    Returns:
        List of Candidate objects with z-scored scores.
    """
    if not candidates:
        return []
    scores = np.array([c.score for c in candidates], dtype=np.float64)
    norm = (scores - scores.mean()) / (scores.std() + MIN_MAX_DELTA)
    return [replace(c, score=float(s)) for c, s in zip(candidates, norm)]


def normalize_scores(
    candidates: List[Candidate], method: str = "min_max", floor: Optional[float] = None
) -> List[Candidate]:
    """
    Normalize candidate scores using the specified method.

    Args:
        candidates: List of Candidate objects.
        method: Normalization strategy ('min_max' or 'z_score').
        floor: Min-max only; see min_max_normalize.

    Returns:
        List of Candidate objects with normalized scores.
    """
    if method == "min_max":
        return min_max_normalize(candidates, floor)
    if method == "z_score":
        return z_score_normalize(candidates)
    raise ValueError(f"Unknown normalization method: {method}")
