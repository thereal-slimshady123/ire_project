"""Experiment 2: does a router beat always-hybrid (alpha tuned on the same training queries) on held-out queries?

Implements the pre-registered protocol in improvements.md §3 exactly:
  * within each dataset, 5-fold CV over queries; alpha*, best single retriever, epsilon (inner 3-fold
    CV on the GBM router) and both routers are fitted on training folds only;
  * primary test: pooled held-out per-query nDCG@10(GBM router) - nDCG@10(always-hybrid), paired
    bootstrap; success iff the 95% CI lower bound > 0;
  * secondary: per dataset, logistic router, best single, random routing with the router's route
    proportions (expected value), oracle (upper bound), leave-one-dataset-out transfer.

Usage:
  python scripts/run_experiment2_heldout_routing.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.evaluation.ranking_metrics import compute_ranking_metrics  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test, paired_wilcoxon_test  # noqa: E402
from router_retrieval.labeling.margin_labeler import MarginLabeler  # noqa: E402
from router_retrieval.retrieval.bm25 import BM25Retriever  # noqa: E402
from router_retrieval.router.gbm_router import GBMRouter  # noqa: E402
from router_retrieval.router.logistic_router import LogisticRouter  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp1 = _load_script("exp1", "run_experiment1_hybrid_degradation.py")
exp1b = _load_script("exp1b", "run_experiment1b_query_types.py")

log = get_logger("experiment2")
PRIMARY = "ndcg_cut_10"
ROUTES = ["bm25", "dense", "hybrid"]
FEATURES = exp1.FEATURE_COLUMNS
DATASETS = ["nfcorpus", "scifact", "fiqa", "cqadupstack-unix", "quora"]
INNER_FOLDS = 3


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2: held-out routing vs tuned always-hybrid")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--quora_queries", type=int, default=1000)
    p.add_argument("--quora_distractors", type=int, default=50000)
    p.add_argument("--n_bootstrap", type=int, default=10000)
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Per-dataset table: per-route nDCG@10 (hybrid over the whole alpha grid) + features
# --------------------------------------------------------------------------------------

def dataset_table(name: str, args, ret_cfg, hyb_cfg, seed: int) -> pd.DataFrame:
    cache = args.output_dir / "cache" / f"{name}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, dtype={"query_id": str})
    if name == "quora":
        path = exp1b.build_quora_subset(args.data_dir / "quora", args.processed_dir / "quora-sub",
                                        args.quora_queries, args.quora_distractors, seed)
    else:
        path = args.data_dir / name
    corpus, queries, qrels = exp1b.load_split(path)
    k = ret_cfg["dense"].get("top_k", 100)
    bm25_run, dense_run = exp1b.get_runs(name, corpus, queries, args, ret_cfg, k)
    alphas = [float(a) for a in hyb_cfg["alpha_grid"]]
    hyb = exp1.build_hybrid_runs(bm25_run, dense_run, alphas, k, hyb_cfg["normalization"]["method"])

    df = pd.DataFrame(index=pd.Index(sorted(queries), name="query_id"))
    df["ndcg_bm25"] = compute_ranking_metrics(qrels, bm25_run, [PRIMARY])[PRIMARY]
    df["ndcg_dense"] = compute_ranking_metrics(qrels, dense_run, [PRIMARY])[PRIMARY]
    for a in alphas:
        df[f"ndcg_hyb_{a}"] = compute_ranking_metrics(qrels, hyb[a], [PRIMARY])[PRIMARY]

    bm25 = BM25Retriever(k1=ret_cfg["bm25"]["k1"], b=ret_cfg["bm25"]["b"])
    bm25.index(corpus)  # IDF statistics for query features
    df = df.join(exp1.compute_features(queries, corpus, bm25, bm25_run, dense_run))
    df.insert(0, "dataset", name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    log.info(f"[{name}] {len(df)} queries | BM25={df.ndcg_bm25.mean():.4f} Dense={df.ndcg_dense.mean():.4f}")
    return df


# --------------------------------------------------------------------------------------
# One train/test fit (used for within-dataset CV folds and for transfer)
# --------------------------------------------------------------------------------------

def route_matrix(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"bm25": df["ndcg_bm25"], "dense": df["ndcg_dense"], "hybrid": df[f"ndcg_hyb_{alpha}"]}, index=df.index
    )


def make_router(kind: str, router_cfg: dict, seed: int):
    if kind == "gbm":
        g = router_cfg["models"]["gbm"]
        return GBMRouter(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"],
                         num_leaves=g["num_leaves"], class_weight=g["class_weight"], random_state=seed)
    lr = router_cfg["models"]["logistic"]
    return LogisticRouter(C=lr["C"], max_iter=lr["max_iter"], class_weight=lr["class_weight"], random_state=seed)


def fit_router(kind, X, R, eps, router_cfg, seed):
    """Fit a router on training features X and training route scores R (labels via margin eps)."""
    y = MarginLabeler(epsilon=eps).label_wide(R)["best_route"]
    weights = (R.max(axis=1) - R.min(axis=1) + 0.01).values  # regret spread
    model = make_router(kind, router_cfg, seed)
    if y.nunique() < 2:  # degenerate fold: constant router
        model.constant = y.iloc[0]
        return model
    model.constant = None
    return model.fit(X, y, sample_weight=weights if kind == "gbm" else None)


def predict_routes(model, X) -> np.ndarray:
    if getattr(model, "constant", None) is not None:
        return np.array([model.constant] * len(X), dtype=object)
    return np.array(model.predict(X), dtype=object)


def routed_scores(R: pd.DataFrame, routes: np.ndarray) -> np.ndarray:
    cols = [R.columns.get_loc(r) for r in routes]
    return R.to_numpy()[np.arange(len(R)), cols]


def tune_epsilon(train: pd.DataFrame, alpha: float, eps_grid, router_cfg, seed) -> float:
    """Inner 3-fold CV on training queries: epsilon maximizing routed nDCG@10 of the GBM router."""
    R = route_matrix(train, alpha)

    def utility(eps):
        total = []
        for itr, ite in create_kfold_splits(list(train.index), INNER_FOLDS, seed):
            m = fit_router("gbm", train.loc[itr, FEATURES], R.loc[itr], eps, router_cfg, seed)
            total.append(routed_scores(R.loc[ite], predict_routes(m, train.loc[ite, FEATURES])))
        return float(np.concatenate(total).mean())

    return MarginLabeler().tune_epsilon(R, eps_grid, utility_fn=utility)


def fit_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, alphas, eps_grid, router_cfg, seed) -> Tuple[pd.DataFrame, dict]:
    """Fit everything on `train` only; score systems per query on `test`."""
    alpha = max(alphas, key=lambda a: train[f"ndcg_hyb_{a}"].mean())
    best_single = "bm25" if train["ndcg_bm25"].mean() >= train["ndcg_dense"].mean() else "dense"
    eps = tune_epsilon(train, alpha, eps_grid, router_cfg, seed)
    R_tr, R_te = route_matrix(train, alpha), route_matrix(test, alpha)

    out = pd.DataFrame(index=test.index)
    out["dataset"] = test["dataset"]
    out["hybrid"] = R_te["hybrid"]
    out["best_single"] = R_te[best_single]
    out["oracle"] = R_te.max(axis=1)
    info = {"alpha": alpha, "best_single": best_single, "epsilon": eps}
    for kind in ("gbm", "logistic"):
        m = fit_router(kind, train[FEATURES], R_tr, eps, router_cfg, seed)
        routes = predict_routes(m, test[FEATURES])
        out[f"router_{kind}"] = routed_scores(R_te, routes)
        out[f"route_{kind}"] = routes
        props = np.array([(routes == r).mean() for r in ROUTES])
        out[f"random_{kind}"] = R_te[ROUTES].to_numpy() @ props  # expected nDCG under random routing
        info[f"route_share_{kind}"] = dict(zip(ROUTES, props.round(4).tolist()))
    return out, info


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def compare(df: pd.DataFrame, a: str, b: str, n_boot: int, seed: int) -> dict:
    boot = paired_bootstrap_test(df[a].values, df[b].values, n_boot, seed)
    wil = paired_wilcoxon_test(df[a].values, df[b].values, alternative="greater")
    return {**boot, "wilcoxon_p_greater": wil["p_value"]}


COMPARISONS = [
    ("router_gbm", "hybrid"),
    ("router_logistic", "hybrid"),
    ("router_gbm", "best_single"),
    ("router_gbm", "random_gbm"),
    ("router_logistic", "random_logistic"),
    ("oracle", "hybrid"),
]


def summarize(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    means = {c: float(df[c].mean()) for c in ("hybrid", "best_single", "router_gbm", "router_logistic",
                                               "random_gbm", "random_logistic", "oracle")}
    return {"n": len(df), "mean": means,
            "tests": {f"{a} - {b}": compare(df, a, b, n_boot, seed) for a, b in COMPARISONS}}


def fmt_ci(t: dict) -> str:
    return f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"


def write_tables(summary: dict, path: Path) -> None:
    prim = summary["pooled"]["tests"]["router_gbm - hybrid"]
    verdict = "SUCCESS" if prim["ci_95_low"] > 0 else "NEGATIVE (CI includes or is below 0)"
    lines = [
        "# Experiment 2 results: held-out routing vs tuned always-hybrid",
        "",
        "Pre-registered in improvements.md §3. All numbers are nDCG@10 on held-out queries (5-fold CV within each "
        "dataset); α, ε, best single retriever and routers fitted on training folds only.",
        "",
        f"**Primary test (pooled, n = {summary['pooled']['n']}): GBM router − always-hybrid = {fmt_ci(prim)}, "
        f"p = {prim['p_value']:.1e} → {verdict}.**",
        "",
        "## Mean held-out nDCG@10",
        "",
        "| Dataset | n | Always-hybrid | Best single | GBM router | Logistic router | Random (GBM mix) | Oracle (bound) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, s in [*summary["per_dataset"].items(), ("**pooled**", summary["pooled"])]:
        m = s["mean"]
        lines.append(
            f"| {name} | {s['n']} | {m['hybrid']:.4f} | {m['best_single']:.4f} | {m['router_gbm']:.4f} "
            f"| {m['router_logistic']:.4f} | {m['random_gbm']:.4f} | {m['oracle']:.4f} |"
        )
    lines += [
        "",
        "## Paired differences [95% bootstrap CI]",
        "",
        "| Dataset | GBM − hybrid | Logistic − hybrid | GBM − best single | GBM − random (same mix) | Oracle − hybrid |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, s in [*summary["per_dataset"].items(), ("**pooled**", summary["pooled"])]:
        t = s["tests"]
        lines.append(
            f"| {name} | {fmt_ci(t['router_gbm - hybrid'])} | {fmt_ci(t['router_logistic - hybrid'])} "
            f"| {fmt_ci(t['router_gbm - best_single'])} | {fmt_ci(t['router_gbm - random_gbm'])} "
            f"| {fmt_ci(t['oracle - hybrid'])} |"
        )
    lines += [
        "",
        "## Route shares chosen by the GBM router (held-out, mean over folds)",
        "",
        "| Dataset | BM25 | Dense | Hybrid | ε chosen per fold | α* per fold |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, folds in summary["fold_info"].items():
        sh = {r: np.mean([f["route_share_gbm"][r] for f in folds]) for r in ROUTES}
        eps_list = ", ".join("{:g}".format(f["epsilon"]) for f in folds)
        alpha_list = ", ".join("{:g}".format(f["alpha"]) for f in folds)
        lines.append(
            f"| {name} | {sh['bm25']:.1%} | {sh['dense']:.1%} | {sh['hybrid']:.1%} | {eps_list} | {alpha_list} |"
        )
    lines += [
        "",
        "## Cross-dataset transfer (train on the other four datasets, test on this one)",
        "",
        "| Target | n | Hybrid (source α) | Hybrid (in-domain CV α) | GBM router | GBM − hybrid(source α) | GBM − hybrid(in-domain α) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, t in summary["transfer"].items():
        lines.append(
            f"| {name} | {t['n']} | {t['mean_hybrid_source_alpha']:.4f} | {t['mean_hybrid_indomain_alpha']:.4f} "
            f"| {t['mean_router_gbm']:.4f} | {fmt_ci(t['gbm_vs_hybrid_source'])} | {fmt_ci(t['gbm_vs_hybrid_indomain'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    ret_cfg = load_config(args.config_dir / "retrievers.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")
    router_cfg = load_config(args.config_dir / "router.yaml")
    eps_grid = [float(e) for e in load_config(args.config_dir / "labels.yaml")["epsilon_grid"]]
    seed, k_folds = exp_cfg["seed"], exp_cfg["k_folds"]
    seed_everything(seed)
    alphas = [float(a) for a in hyb_cfg["alpha_grid"]]

    tables = {ds: dataset_table(ds, args, ret_cfg, hyb_cfg, seed) for ds in args.datasets}

    # Within-dataset held-out evaluation.
    heldout, fold_info = [], {}
    for ds, df in tables.items():
        fold_info[ds] = []
        for i, (train_ids, test_ids) in enumerate(create_kfold_splits(list(df.index), k_folds, seed)):
            out, info = fit_and_evaluate(df.loc[train_ids], df.loc[test_ids], alphas, eps_grid, router_cfg, seed)
            out["fold"] = i
            heldout.append(out)
            fold_info[ds].append(info)
        log.info(f"[{ds}] folds done: " + "; ".join(f"α={f['alpha']:g} ε={f['epsilon']:g}" for f in fold_info[ds]))
    heldout = pd.concat(heldout)
    heldout.index = [f"{d}:{q}" for d, q in zip(heldout["dataset"], heldout.index)]

    # Leave-one-dataset-out transfer (secondary).
    transfer = {}
    if len(tables) > 1:
        for target in tables:
            source = pd.concat([t for n, t in tables.items() if n != target])
            source.index = [f"{d}:{q}" for d, q in zip(source["dataset"], source.index)]
            out, info = fit_and_evaluate(source, tables[target], alphas, eps_grid, router_cfg, seed)
            out_ids = [f"{target}:{q}" for q in out.index]
            indomain = heldout.loc[out_ids, "hybrid"].values
            transfer[target] = {
                "n": len(out), **info,
                "mean_hybrid_source_alpha": float(out["hybrid"].mean()),
                "mean_hybrid_indomain_alpha": float(indomain.mean()),
                "mean_router_gbm": float(out["router_gbm"].mean()),
                "gbm_vs_hybrid_source": paired_bootstrap_test(out["router_gbm"].values, out["hybrid"].values,
                                                              args.n_bootstrap, seed),
                "gbm_vs_hybrid_indomain": paired_bootstrap_test(out["router_gbm"].values, indomain,
                                                                args.n_bootstrap, seed),
            }
            log.info(f"[transfer → {target}] α={info['alpha']:g} ε={info['epsilon']:g} "
                     f"GBM−hybrid(src α)={transfer[target]['gbm_vs_hybrid_source']['delta_mean']:+.4f}")

    summary = {
        "preregistration": "improvements.md §3",
        "dense_model": args.dense_model,
        "pooled": summarize(heldout, args.n_bootstrap, seed),
        "per_dataset": {ds: summarize(heldout[heldout["dataset"] == ds], args.n_bootstrap, seed) for ds in tables},
        "fold_info": fold_info,
        "transfer": transfer,
    }
    prim = summary["pooled"]["tests"]["router_gbm - hybrid"]
    summary["primary_verdict"] = "success" if prim["ci_95_low"] > 0 else "negative"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(args.output_dir / "per_query_heldout.csv")
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    write_tables(summary, args.output_dir / "tables.md")
    log.info(f"PRIMARY: GBM router − always-hybrid (pooled n={len(heldout)}) = {fmt_ci(prim)} "
             f"p={prim['p_value']:.1e} → {summary['primary_verdict'].upper()}")
    log.info(f"Wrote {args.output_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
