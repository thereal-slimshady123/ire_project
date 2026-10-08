"""Abstract base class for query routers."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import pandas as pd


class RouterModel(ABC):
    """Abstract interface for multi-class query routers."""

    def __init__(self, classes: Optional[List[str]] = None):
        self.classes = classes or ["bm25", "dense", "hybrid"]

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "RouterModel":
        """
        Fit router on training features and route labels.

        Args:
            X: Feature matrix.
            y: Target route labels.
            sample_weight: Optional per-sample weights (e.g., regret weights).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability distribution over routes.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Numpy array of shape (n_samples, n_classes) with class probabilities.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> List[str]:
        """
        Predict discrete route selection (argmax over classes).

        Args:
            X: Feature matrix.

        Returns:
            List of predicted route names.
        """
        pass
