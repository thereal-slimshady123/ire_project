"""Gradient boosted routers: a 3-class classifier (regret weighting) and the Dual Advantage Regressor."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# ----------------------------------------------------------------------------------------------
# Dual Advantage Regression (improvements.md section 2.2)
# ----------------------------------------------------------------------------------------------

ROUTES = ("bm25", "dense", "hybrid")


def route_label(delta_bm25: np.ndarray, delta_dense: np.ndarray, epsilon: float = 0.02) -> np.ndarray:
    """
    Best route per query from the true advantages over hybrid, hybrid unless a margin is cleared.

    BM25 / dense is the label only if its advantage is >= epsilon and strictly the larger of the two.
    """
    b, d = np.asarray(delta_bm25, dtype=float), np.asarray(delta_dense, dtype=float)
    out = np.full(len(b), "hybrid", dtype=object)
    out[(b >= epsilon) & (b > d)] = "bm25"
    out[(d >= epsilon) & (d > b)] = "dense"
    return out


def decide_route(
    delta_pred: np.ndarray, theta_bm25: float, theta_dense: float
) -> np.ndarray:
    """
    Risk-thresholded decision rule: divert from hybrid only when the predicted gain clears theta.

    BM25 if D_b > theta_bm25 and D_b > D_d; dense if D_d > theta_dense and D_d > D_b; otherwise hybrid
    (the safe default for marginal, tied or uncertain predictions).
    """
    delta_pred = np.asarray(delta_pred, dtype=float).reshape(-1, 2)
    b, d = delta_pred[:, 0], delta_pred[:, 1]
    out = np.full(len(b), "hybrid", dtype=object)
    out[(b > theta_bm25) & (b > d)] = "bm25"
    out[(d > theta_dense) & (d > b)] = "dense"
    return out


def routed_gain(delta: np.ndarray, routes: np.ndarray) -> np.ndarray:
    """nDCG gain over hybrid of each routing decision: Delta_bm25, Delta_dense, or 0 for hybrid."""
    delta = np.asarray(delta, dtype=float).reshape(-1, 2)
    return np.where(routes == "bm25", delta[:, 0], np.where(routes == "dense", delta[:, 1], 0.0))


class _ConstantRegressor:
    """Predicts the (weighted) training mean; used when there are too few rows to fit a tree model."""

    def __init__(self, value: float = 0.0):
        self.value = float(value)

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.value)


class AdvantageRouter:
    """
    Dual Advantage Regression router.

    Two regressors predict the continuous margins over the hybrid baseline,
        D_bm25(q)  = nDCG@10(BM25, q)  - nDCG@10(Hybrid, q)
        D_dense(q) = nDCG@10(Dense, q) - nDCG@10(Hybrid, q),
    and a risk threshold per route decides whether the predicted margin is large enough to leave the
    hybrid default (`decide_route`). On `fit`, the thresholds are calibrated on out-of-fold predictions
    from an inner K-fold over the training queries: the pair (theta_bm25, theta_dense) maximizing mean
    routed gain wins (ties -> the more conservative, i.e. larger, pair). If that gain is not positive and
    significant (z = mean / s.e. >= `min_gain_z`, guarding against selection bias from searching the
    grid), both thresholds become +inf and the router degenerates to always-hybrid.

    `predict_proba` is a softmax over the predicted advantages [D_bm25, D_dense, 0 (hybrid)] at a
    temperature fitted on the same out-of-fold predictions (NLL against the margin labels), giving the
    confidence scores that tau-escalation consumes. Note that `predict` follows the theta rule, not the
    argmax of `predict_proba`.
    """

    classes = list(ROUTES)

    def __init__(
        self,
        backend: str = "lightgbm",
        n_estimators: int = 150,
        learning_rate: float = 0.03,
        num_leaves: int = 7,
        min_child_samples: int = 30,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        objective: str = "regression",
        random_state: int = 42,
        theta_grid: Sequence[float] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10),
        theta_bm25: float = 0.02,
        theta_dense: float = 0.02,
        label_epsilon: float = 0.02,
        inner_folds: int = 3,
        min_gain_z: float = 1.645,
        n_jobs: int = 4,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        if backend not in ("lightgbm", "hist_gbm"):
            raise ValueError(f"Unsupported backend: {backend}")
        self.backend = backend
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.random_state = random_state
        self.theta_grid = [float(t) for t in theta_grid]
        self.theta_bm25 = float(theta_bm25)
        self.theta_dense = float(theta_dense)
        self.label_epsilon = label_epsilon
        self.inner_folds = inner_folds
        self.min_gain_z = min_gain_z
        self.n_jobs = n_jobs
        self.extra_params = extra_params or {}
        self.models: Optional[Tuple[Any, Any]] = None
        self.temperature = 0.05
        self.feature_names_: List[str] = []
        self.oof_gain_: Optional[float] = None
        self.oof_z_: Optional[float] = None
        self.calibrated_ = False

    # ---------------------------------------------------------------------------- regressors

    def _make_regressor(self):
        if self.backend == "hist_gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                max_iter=self.n_estimators, learning_rate=self.learning_rate, max_leaf_nodes=self.num_leaves,
                min_samples_leaf=self.min_child_samples, l2_regularization=self.reg_lambda,
                random_state=self.random_state, **self.extra_params,
            )
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            objective=self.objective, n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples, subsample=self.subsample,
            subsample_freq=1, colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
            random_state=self.random_state, n_jobs=self.n_jobs, deterministic=True, force_row_wise=True,
            verbose=-1, **self.extra_params,
        )

    def _fit_pair(self, X, d_b, d_d, w) -> Tuple[Any, Any]:
        models = []
        for y in (d_b, d_d):
            if len(y) < 2 * max(1, min(self.min_child_samples, 2)):  # too few rows for any split
                models.append(_ConstantRegressor(np.average(y, weights=w) if len(y) else 0.0))
                continue
            m = self._make_regressor()
            m.fit(X, y, sample_weight=w)
            models.append(m)
        return models[0], models[1]

    @staticmethod
    def _predict_pair(models, X) -> np.ndarray:
        return np.column_stack([models[0].predict(X), models[1].predict(X)])

    # ------------------------------------------------------------------------------- fitting

    def fit(
        self,
        X: pd.DataFrame,
        delta_bm25: Sequence[float],
        delta_dense: Sequence[float],
        sample_weight: Optional[np.ndarray] = None,
        eval_mask: Optional[np.ndarray] = None,
        calibrate: bool = True,
    ) -> "AdvantageRouter":
        """
        Fit both regressors and (optionally) calibrate the risk thresholds and the proba temperature.

        Args:
            X: Training feature matrix.
            delta_bm25, delta_dense: True advantages over hybrid for each training row.
            sample_weight: Optional per-row weights (e.g. down-weighting synthetic queries).
            eval_mask: Rows eligible as inner held-out for calibration (default: all). Rows outside the
                mask (e.g. synthetic queries) are always in the inner training side and never scored.
            calibrate: Calibrate theta and temperature; otherwise keep the configured thresholds.
        """
        self.feature_names_ = list(X.columns)
        d_b = np.asarray(delta_bm25, dtype=float)
        d_d = np.asarray(delta_dense, dtype=float)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        Xv = X.to_numpy(dtype=float)
        self.models = self._fit_pair(Xv, d_b, d_d, w)
        self.calibrated_ = False
        if calibrate:
            self._calibrate(Xv, d_b, d_d, w, np.ones(len(Xv), bool) if eval_mask is None else np.asarray(eval_mask))
        return self

    def _calibrate(self, X, d_b, d_d, w, eval_mask) -> None:
        eval_idx = np.flatnonzero(eval_mask)
        fixed_idx = np.flatnonzero(~eval_mask)
        k = min(self.inner_folds, len(eval_idx))
        if k < 2:  # too few queries to calibrate: stay conservative
            self.theta_bm25 = self.theta_dense = float("inf")
            self.oof_gain_ = 0.0
            return
        perm = np.random.default_rng(self.random_state).permutation(eval_idx)
        oof = np.zeros((len(X), 2))
        for held in np.array_split(perm, k):
            train = np.concatenate([fixed_idx, np.setdiff1d(eval_idx, held)])
            models = self._fit_pair(X[train], d_b[train], d_d[train], None if w is None else w[train])
            oof[held] = self._predict_pair(models, X[held])
        pred, delta = oof[eval_idx], np.column_stack([d_b[eval_idx], d_d[eval_idx]])

        best = (-np.inf, 0.0, 0.0)  # (gain, theta_b, theta_d)
        for tb in self.theta_grid:
            for td in self.theta_grid:
                gain = float(routed_gain(delta, decide_route(pred, tb, td)).mean())
                if gain > best[0] + 1e-12 or (abs(gain - best[0]) <= 1e-12 and tb + td > best[1] + best[2]):
                    best = (gain, tb, td)
        self.oof_gain_ = best[0]
        # Picking the best of |grid|^2 pairs on the same out-of-fold predictions is optimistic, so the gain
        # must also be statistically distinguishable from zero (z = mean / standard error) to leave hybrid.
        g = routed_gain(delta, decide_route(pred, best[1], best[2]))
        se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0
        self.oof_z_ = float(best[0] / se) if se > 0 else (np.inf if best[0] > 0 else 0.0)
        if best[0] <= 0.0 or self.oof_z_ < self.min_gain_z:
            self.theta_bm25 = self.theta_dense = float("inf")
        else:
            self.theta_bm25, self.theta_dense = best[1], best[2]

        labels = route_label(delta[:, 0], delta[:, 1], self.label_epsilon)
        y = np.array([self.classes.index(l) for l in labels])
        best_nll = (np.inf, self.temperature)
        for temp in np.logspace(-2.5, 0.0, 26):
            p = self._softmax(pred, temp)
            nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())
            if nll < best_nll[0]:
                best_nll = (nll, float(temp))
        self.temperature = best_nll[1]
        self.calibrated_ = True

    # ----------------------------------------------------------------------------- prediction

    @staticmethod
    def _softmax(delta_pred: np.ndarray, temperature: float) -> np.ndarray:
        z = np.column_stack([delta_pred[:, 0], delta_pred[:, 1], np.zeros(len(delta_pred))]) / temperature
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_names_ or list(X.columns)
        return X[cols].to_numpy(dtype=float)

    def predict_delta(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted advantages, shape (n, 2): columns [D_bm25_hat, D_dense_hat]."""
        return self._predict_pair(self.models, self._matrix(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Confidence over routes [bm25, dense, hybrid]: softmax of [D_bm25_hat, D_dense_hat, 0] / T."""
        return self._softmax(self.predict_delta(X), self.temperature)

    def predict(self, X: pd.DataFrame) -> List[str]:
        """Route per query under the calibrated risk-threshold rule."""
        return decide_route(self.predict_delta(X), self.theta_bm25, self.theta_dense).tolist()

    def feature_importances(self) -> Optional[pd.DataFrame]:
        """Gain-based importances (normalized to sum to 1) for each regressor; None for backends without them."""
        if self.backend != "lightgbm" or self.models is None or any(
                isinstance(m, _ConstantRegressor) for m in self.models):
            return None
        cols = {}
        for name, m in zip(("delta_bm25", "delta_dense"), self.models):
            g = m.booster_.feature_importance(importance_type="gain").astype(float)
            cols[name] = g / g.sum() if g.sum() > 0 else g
        return pd.DataFrame(cols, index=self.feature_names_)

    def save(self, path: Path) -> None:
        """Persist model artifact."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "AdvantageRouter":
        """Load model artifact."""
        with open(path, "rb") as f:
            return pickle.load(f)
