"""Cross-validated router training / evaluation on route tables (shared by Track 2 scripts).

A *route table* has one row per query: per-route nDCG@10 (`ndcg_bm25`, `ndcg_dense`, `ndcg_hyb_<alpha>`)
plus feature columns. Everything that depends on outcomes (alpha*, route labels, risk thresholds,
temperature, regressors) is fitted on the training rows of a fold only; held-out rows contribute their
features and, afterwards, their scores for evaluation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from router_retrieval.data.splits import create_kfold_splits
from router_retrieval.evaluation.significance import paired_bootstrap_test, paired_wilcoxon_test
from router_retrieval.features.feature_builder import (
    FEATURE_COLUMNS,
    QUERY_FEATURES,
    RETRIEVAL_SIGNAL_FEATURES,
)
from router_retrieval.labeling.margin_labeler import MarginLabeler
from router_retrieval.router.advantage_targets import (
    advantage_targets,
    hybrid_column,
    select_alpha,
    stability_mask,
)
from router_retrieval.router.gbm_router import AdvantageRouter, GBMRouter, decide_route
from router_retrieval.router.logistic_router import LogisticRouter

ROUTES = ["bm25", "dense", "hybrid"]
# advantage: calibrated thetas + significance guard (deployable); advantage_fixed: thetas fixed at the config
# defaults, no calibration (exposes the raw regressors' signal); *_cls: 3-class baselines.
KINDS = ("advantage", "advantage_fixed", "gbm_cls", "logistic_cls")
ADVANTAGE_KINDS = ("advantage", "advantage_fixed")

FEATURE_GROUPS: Dict[str, List[str]] = {
    "all": FEATURE_COLUMNS,
    "query_only": QUERY_FEATURES,
    "retrieval_signal": RETRIEVAL_SIGNAL_FEATURES,
    # Everything available before the dense retriever has run (query + BM25-side signals): the
    # feature set a latency-saving router (issue A7) could actually use.
    "pre_dense": QUERY_FEATURES + [f for f in RETRIEVAL_SIGNAL_FEATURES if f.startswith("bm25_")],
}


# ------------------------------------------------------------------------------------ data

def load_route_table(dataset: str, results_dir: Path = Path("results"), use_features_parquet: bool = False) -> pd.DataFrame:
    """
    Per-query route table of a dataset from the committed Experiment 2 cache.

    With `use_features_parquet`, the feature columns of `results/features/{dataset}.parquet` (Contract 2,
    float32) replace the cached ones when that file covers every query, so freshly extracted features
    flow into training. Off by default: the committed results are reproduced exactly from the cache.
    """
    path = Path(results_dir) / "experiment2" / "cache" / f"{dataset}.csv"
    df = pd.read_csv(path, index_col=0, dtype={"query_id": str})
    df.index = df.index.astype(str)
    df.index.name = "query_id"
    feats = Path(results_dir) / "features" / f"{dataset}.parquet"
    if use_features_parquet and feats.exists():
        fdf = pd.read_parquet(feats)
        fdf["query_id"] = fdf["query_id"].astype(str)
        fdf = fdf.set_index("query_id")
        cols = [c for c in FEATURE_COLUMNS if c in fdf.columns]
        if set(df.index) <= set(fdf.index) and cols:
            df[cols] = fdf.loc[df.index, cols].astype(float).to_numpy()
    return df


def prefix_index(table: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = table.copy()
    out.index = [f"{dataset}:{q}" for q in table.index]
    return out


def advantage_kwargs(router_cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """AdvantageRouter constructor arguments from configs/router.yaml (`advantage` section)."""
    cfg = dict(router_cfg.get("advantage", {}))
    cfg.pop("stability", None)
    cfg.pop("synthetic_weight", None)
    cfg.setdefault("random_state", seed)
    if "theta_grid" in cfg:
        cfg["theta_grid"] = [float(t) for t in cfg["theta_grid"]]
    return cfg


# --------------------------------------------------------------------------- fold fitting

def _routed_scores(R: pd.DataFrame, routes: np.ndarray) -> np.ndarray:
    cols = [R.columns.get_loc(r) for r in routes]
    return R.to_numpy()[np.arange(len(R)), cols]


def _classifier(kind: str, router_cfg: Dict[str, Any], seed: int):
    if kind == "gbm_cls":
        g = router_cfg["models"]["gbm"]
        return GBMRouter(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"],
                         num_leaves=g["num_leaves"], class_weight=g["class_weight"], random_state=seed)
    lr = router_cfg["models"]["logistic"]
    return LogisticRouter(C=lr["C"], max_iter=lr["max_iter"], class_weight=lr["class_weight"], random_state=seed)


def fit_predict_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kind: str,
    feature_cols: Sequence[str],
    alphas: Sequence[float],
    router_cfg: Dict[str, Any],
    seed: int = 42,
    synthetic: Optional[pd.DataFrame] = None,
    synthetic_weight: Optional[float] = None,
    stability: Optional[bool] = None,
    label_epsilon: float = 0.02,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit a router on `train` (plus optional `synthetic` rows) and score it per query on `test`.

    Returns a per-query frame (hybrid / bm25 / dense / routed / oracle / random-same-mix nDCG@10, the chosen
    route, Contract-3 probabilities and advantage estimates) and a dict describing the fit.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    feature_cols = list(feature_cols)
    alpha = select_alpha(train, alphas)  # natural training queries only
    R_te = pd.DataFrame({"bm25": test["ndcg_bm25"], "dense": test["ndcg_dense"],
                         "hybrid": test[hybrid_column(alpha)]}, index=test.index)
    out = pd.DataFrame(index=test.index)
    out["hybrid"] = R_te["hybrid"]
    out["bm25"], out["dense"] = R_te["bm25"], R_te["dense"]
    out["oracle"] = R_te.max(axis=1)
    info: Dict[str, Any] = {"kind": kind, "alpha": alpha, "n_train": len(train),
                            "n_synthetic": 0 if synthetic is None else len(synthetic)}

    if kind in ADVANTAGE_KINDS:
        adv_cfg = router_cfg.get("advantage", {})
        stab_cfg = adv_cfg.get("stability", {})
        use_stability = stab_cfg.get("enabled", False) if stability is None else stability
        w_syn = adv_cfg.get("synthetic_weight", 1.0) if synthetic_weight is None else synthetic_weight
        parts, weights, is_nat = [train], [np.ones(len(train))], [np.ones(len(train), bool)]
        if synthetic is not None and len(synthetic):
            parts.append(synthetic)
            weights.append(np.full(len(synthetic), float(w_syn)))
            is_nat.append(np.zeros(len(synthetic), bool))
        fit_tab = pd.concat(parts)
        w, nat = np.concatenate(weights), np.concatenate(is_nat)
        keep = np.ones(len(fit_tab), bool)
        if use_stability:
            keep = stability_mask(fit_tab, alpha, alphas, stab_cfg.get("alpha_shift", 0.1),
                                  stab_cfg.get("epsilon", 0.02)).to_numpy()
            info["stability_dropped"] = float(1.0 - keep.mean())
        tgt = advantage_targets(fit_tab, alpha)
        router = AdvantageRouter(**advantage_kwargs(router_cfg, seed))
        router.fit(fit_tab[feature_cols].iloc[np.flatnonzero(keep)],
                   tgt["delta_bm25"].to_numpy()[keep], tgt["delta_dense"].to_numpy()[keep],
                   sample_weight=w[keep], eval_mask=nat[keep], calibrate=(kind == "advantage"))
        d_pred = router.predict_delta(test[feature_cols])
        routes = decide_route(d_pred, router.theta_bm25, router.theta_dense)
        proba = router.predict_proba(test[feature_cols])
        out["delta_pred_bm25"], out["delta_pred_dense"] = d_pred[:, 0], d_pred[:, 1]
        info.update(theta_bm25=router.theta_bm25, theta_dense=router.theta_dense,
                    temperature=router.temperature, oof_gain=router.oof_gain_,
                    n_fit_rows=int(keep.sum()))
        imp = router.feature_importances()
        if imp is not None:
            info["importances"] = imp.to_dict()
    else:
        if synthetic is not None and len(synthetic):
            raise ValueError("synthetic training rows are only supported for the advantage kinds")
        R_tr = pd.DataFrame({"bm25": train["ndcg_bm25"], "dense": train["ndcg_dense"],
                             "hybrid": train[hybrid_column(alpha)]}, index=train.index)
        y = MarginLabeler(epsilon=label_epsilon).label_wide(R_tr)["best_route"]
        if y.nunique() < 2:  # degenerate fold: constant router
            routes = np.array([y.iloc[0]] * len(test), dtype=object)
            proba = np.tile([float(c == y.iloc[0]) for c in ROUTES], (len(test), 1))
        else:
            model = _classifier(kind, router_cfg, seed)
            weights = (R_tr.max(axis=1) - R_tr.min(axis=1) + 0.01).to_numpy() if kind == "gbm_cls" else None
            model.fit(train[feature_cols], y, sample_weight=weights)
            proba = model.predict_proba(test[feature_cols])
            routes = np.array(model.predict(test[feature_cols]), dtype=object)
        out["delta_pred_bm25"] = out["delta_pred_dense"] = np.nan

    out["route_predicted"] = routes
    out["routed"] = _routed_scores(R_te, routes)
    out[["p_bm25", "p_dense", "p_hybrid"]] = proba
    props = np.array([(routes == r).mean() for r in ROUTES])
    out["random_same_mix"] = R_te[ROUTES].to_numpy() @ props  # expected nDCG under random routing
    info["route_share"] = dict(zip(ROUTES, props.round(4).tolist()))
    return out, info


def cv_predict(
    table: pd.DataFrame,
    kind: str,
    feature_cols: Sequence[str],
    alphas: Sequence[float],
    router_cfg: Dict[str, Any],
    k_folds: int = 5,
    seed: int = 42,
    synthetic: Optional[pd.DataFrame] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Held-out predictions for every query of `table` via k-fold CV (folds match Experiment 2's)."""
    frames, infos = [], []
    for i, (train_ids, test_ids) in enumerate(create_kfold_splits(list(table.index), k_folds, seed)):
        out, info = fit_predict_fold(table.loc[train_ids], table.loc[test_ids], kind, feature_cols, alphas,
                                     router_cfg, seed, synthetic=synthetic, **kwargs)
        out["fold"] = i
        info["fold"] = i
        frames.append(out)
        infos.append(info)
    held = pd.concat(frames).loc[table.index]
    return held, infos


# ----------------------------------------------------------------------------- reporting

def summarize_heldout(held: pd.DataFrame, n_bootstrap: int = 10000, seed: int = 42) -> Dict[str, Any]:
    """Means and paired tests (routed vs hybrid / random-same-mix, oracle vs hybrid) on held-out rows."""
    boot = lambda a, b: paired_bootstrap_test(held[a].to_numpy(), held[b].to_numpy(), n_bootstrap, seed)
    tests = {"routed - hybrid": boot("routed", "hybrid"),
             "routed - random_same_mix": boot("routed", "random_same_mix"),
             "oracle - hybrid": boot("oracle", "hybrid")}
    tests["routed - hybrid"]["wilcoxon_p_two_sided"] = paired_wilcoxon_test(
        held["routed"].to_numpy(), held["hybrid"].to_numpy())["p_value"]
    gain = (held["routed"] - held["hybrid"]).to_numpy()
    return {
        "n": int(len(held)),
        "mean": {c: float(held[c].mean()) for c in ("hybrid", "bm25", "dense", "routed", "random_same_mix", "oracle")},
        "route_share": {r: float((held["route_predicted"] == r).mean()) for r in ROUTES},
        "wins_losses_ties": {"win": int((gain > 1e-9).sum()), "loss": int((gain < -1e-9).sum()),
                             "tie": int((np.abs(gain) <= 1e-9).sum())},
        "headroom_captured": (float(gain.mean() / (held["oracle"] - held["hybrid"]).mean())
                              if (held["oracle"] - held["hybrid"]).mean() > 0 else float("nan")),
        "tests": tests,
    }


def fmt_ci(t: Dict[str, float]) -> str:
    return f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"


def predictions_frame(held: pd.DataFrame) -> pd.DataFrame:
    """Contract 3 (`results/models/{dataset}/predictions.parquet`) from held-out predictions."""
    out = held.rename_axis("query_id").reset_index()
    out["query_id"] = out["query_id"].astype(str)
    cols = ["query_id", "fold", "route_predicted", "p_bm25", "p_dense", "p_hybrid",
            "delta_pred_bm25", "delta_pred_dense"]
    out = out[cols].copy()
    out["fold"] = out["fold"].astype(np.int32)
    for c in cols[3:]:
        out[c] = out[c].astype(np.float32)
    return out
