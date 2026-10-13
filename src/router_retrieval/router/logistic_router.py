"""Logistic Regression baseline router."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from router_retrieval.router.base import RouterModel


class LogisticRouter(RouterModel):
    """
    Logistic regression classifier for 3-class route prediction.
    Features are standardized inside the model. Class balancing and regret weighting both come
    from the sample_weight passed to fit() (labeling.regret.compute_sample_weights).
    """

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        super().__init__(classes=classes)
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LogisticRouter":
        """Fit logistic regression router with per-sample weights."""
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
    def load(cls, path: Path) -> "LogisticRouter":
        """Load model artifact."""
        raise NotImplementedError
