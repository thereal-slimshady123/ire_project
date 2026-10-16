"""Gradient Boosted Trees (LightGBM/XGBoost) router with regret weighting."""

import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from router_retrieval.router.base import RouterModel
from router_retrieval.router.logistic_router import _align_proba


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
        if self.backend != "lightgbm":
            raise ValueError(f"Unsupported backend: {self.backend}")
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            class_weight=self.class_weight,
            random_state=self.random_state,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
            **self.extra_params,
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probability matrix with columns ordered as `self.classes`."""
        return _align_proba(self.model.predict_proba(X), list(self.model.classes_), self.classes)

    def predict(self, X: pd.DataFrame) -> List[str]:
        """Predict route for each sample."""
        return [self.classes[i] for i in self.predict_proba(X).argmax(axis=1)]

    def save(self, path: Path) -> None:
        """Persist model artifact."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "GBMRouter":
        """Load model artifact."""
        with open(path, "rb") as f:
            return pickle.load(f)
