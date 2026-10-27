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
