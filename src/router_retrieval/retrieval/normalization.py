"""Score normalization utilities for sparse and dense retrieval scores."""

from typing import List
from router_retrieval.data.schema import Candidate


def min_max_normalize(candidates: List[Candidate]) -> List[Candidate]:
    """
    Min-max normalize candidate scores to [0, 1] range.

    Args:
        candidates: List of Candidate objects for a query.

    Returns:
        List of Candidate objects with normalized scores.
    """
    raise NotImplementedError


def z_score_normalize(candidates: List[Candidate]) -> List[Candidate]:
    """
    Z-score normalize candidate scores (zero mean, unit variance).

    Args:
        candidates: List of Candidate objects for a query.

    Returns:
        List of Candidate objects with z-scored scores.
    """
    raise NotImplementedError


def normalize_scores(candidates: List[Candidate], method: str = "min_max") -> List[Candidate]:
    """
    Normalize candidate scores using the specified method.

    Args:
        candidates: List of Candidate objects.
        method: Normalization strategy ('min_max' or 'z_score').

    Returns:
        List of Candidate objects with normalized scores.
    """
    raise NotImplementedError
