"""Escalation decision logic based on prediction confidence threshold tau."""

from typing import List, Tuple
import numpy as np


class EscalationDecider:
    """
    Applies confidence threshold tau to class probability distribution.
    If max(proba) < tau -> ESCALATE
    Else -> route with argmax(proba)
    """

    def __init__(self, tau: float = 0.5, classes: List[str] = None):
        """
        Initialize EscalationDecider.

        Args:
            tau: Confidence threshold for escalation.
            classes: Ordered list of route class names.
        """
        self.tau = tau
        self.classes = classes or ["bm25", "dense", "hybrid"]

    def decide(self, probas: np.ndarray) -> Tuple[List[str], List[bool], np.ndarray]:
        """
        Decide routing or escalation for each query.

        Args:
            probas: Array of shape (n_queries, n_classes).

        Returns:
            Tuple of:
                - decisions: List of route strings or 'escalate'
                - is_escalated: List of booleans indicating escalation
                - confidence_scores: Array of max probability per query
        """
        raise NotImplementedError

    def find_tau_for_target_rate(
        self,
        probas: np.ndarray,
        target_escalation_rate: float,
    ) -> float:
        """
        Calibrate tau on training set to achieve a target escalation rate (e.g. 5%, 10%, 15%).

        Args:
            probas: Training predicted probabilities.
            target_escalation_rate: Target fraction of queries to escalate.

        Returns:
            Calibrated tau threshold.
        """
        raise NotImplementedError
