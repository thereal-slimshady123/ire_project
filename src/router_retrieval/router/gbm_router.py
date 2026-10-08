"""Gradient Boosted Trees (LightGBM/XGBoost) router with regret weighting."""

from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from router_retrieval.router.base import RouterModel


class GBMRouter(RouterModel):
    """Gradient boosted decision tree router supporting regret-weighted sample weights."""

    def __init__(
        self,
        backend: str = "lightgbm",
        classes: Optional[List[str]] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(classes=classes)
        self.backend = backend
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.class_weight = class_weight
        self.extra_params = extra_params or {}
        self.model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "GBMRouter":
        """
        Fit gradient boosted tree model.

        Args:
            X: Feature matrix.
            y: Target route labels.
            sample_weight: Per-sample regret weights.
        """
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probability matrix."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> List[str]:
        """Predict route for each sample."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Persist model artifact."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "GBMRouter":
        """Load model artifact."""
        raise NotImplementedError
