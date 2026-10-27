"""Router classification diagnostic metrics."""

from typing import Dict, List, Any
import numpy as np


def compute_router_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None,
) -> Dict[str, Any]:
    """
    Compute macro-F1, per-class precision/recall, and confusion matrix.

    Args:
        y_true: Ground truth best routes.
        y_pred: Router predicted routes.
        labels: Class names list.

    Returns:
        Dictionary containing:
            - 'macro_f1': float
            - 'per_class': Dict[str, Dict[str, float]]
            - 'confusion_matrix': np.ndarray
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    labels = labels or sorted(set(y_true) | set(y_pred))
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {"macro_f1": float(np.mean(f)) if len(f) else 0.0,
            "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(support[i])}
                          for i, label in enumerate(labels)},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "labels": list(labels)}


def compute_win_loss_tie_counts(scores_a, scores_b, tie_tolerance: float = 1e-12) -> Dict[str, int]:
    """Count paired per-query wins, losses, and ties for system A against B."""
    a, b = np.asarray(scores_a, dtype=np.float64), np.asarray(scores_b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("scores_a and scores_b must be 1-D arrays of equal length")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("scores must be finite")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    delta = a - b
    return {"wins": int(np.count_nonzero(delta > tie_tolerance)),
            "losses": int(np.count_nonzero(delta < -tie_tolerance)),
            "ties": int(np.count_nonzero(np.abs(delta) <= tie_tolerance)),
            "n": int(len(delta))}
