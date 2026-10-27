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
        p = np.asarray(probas, dtype=float)
        if p.ndim != 2 or p.shape[1] != len(self.classes):
            raise ValueError("probas must have shape (n_queries, n_classes)")
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("probabilities must be finite and non-negative")
        totals = p.sum(axis=1)
        if not np.allclose(totals, 1.0, atol=1e-6):
            raise ValueError("each probability row must sum to 1")
        conf = p.max(axis=1)
        best = p.argmax(axis=1)
        escalated = conf < self.tau
        decisions = ["escalate" if e else self.classes[j] for e, j in zip(escalated, best)]
        return decisions, escalated.tolist(), conf

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
        p = np.asarray(probas, dtype=float)
        if p.ndim != 2 or p.shape[1] != len(self.classes) or len(p) == 0:
            raise ValueError("probas must be a non-empty matrix matching the configured classes")
        if not 0.0 <= target_escalation_rate <= 1.0:
            raise ValueError("target_escalation_rate must be in [0, 1]")
        confidence = p.max(axis=1)
        # Strict '< tau' means tied confidence values cannot be split. Choose the
        # attainable rate closest to the target; on equal distance prefer fewer calls.
        if target_escalation_rate == 0:
            return float(confidence.min())
        if target_escalation_rate == 1:
            return float(np.nextafter(confidence.max(), np.inf))
        thresholds = [float(confidence.min())]
        thresholds.extend(float(np.nextafter(v, np.inf)) for v in np.unique(confidence))
        rates = [float((confidence < t).mean()) for t in thresholds]
        best = min(range(len(thresholds)), key=lambda i: (abs(rates[i] - target_escalation_rate), rates[i]))
        return thresholds[best]
