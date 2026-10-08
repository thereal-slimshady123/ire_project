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
    raise NotImplementedError
