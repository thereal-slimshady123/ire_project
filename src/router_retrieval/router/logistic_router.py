"""Logistic Regression baseline router."""

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from router_retrieval.router.base import RouterModel


class LogisticRouter(RouterModel):
    """Logistic regression classifier for 3-class route prediction."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        super().__init__(classes=classes)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LogisticRouter":
        """Fit logistic regression router (features standardized inside the pipeline)."""
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.C, max_iter=self.max_iter, class_weight=self.class_weight, random_state=self.random_state
            ),
        )
        self.model.fit(X, y, logisticregression__sample_weight=sample_weight)
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
    def load(cls, path: Path) -> "LogisticRouter":
        """Load model artifact."""
        with open(path, "rb") as f:
            return pickle.load(f)


def _align_proba(proba: np.ndarray, fitted: List[str], wanted: List[str]) -> np.ndarray:
    """Reorder/expand probability columns to `wanted`; classes absent from training get 0."""
    out = np.zeros((proba.shape[0], len(wanted)))
    for j, c in enumerate(fitted):
        out[:, wanted.index(c)] = proba[:, j]
    return out
