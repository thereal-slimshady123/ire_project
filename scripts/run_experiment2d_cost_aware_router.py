"""Experiment 2d (exploratory, pre-declared): a cost-aware router vs tuned always-hybrid.

Design fixed in results/preregistration/experiment2d_cost_aware_router_20261031.md (hash in SHA256SUMS.txt):
  * Experiment 2's datasets, folds and training-fold alpha*;
  * primary router = Experiment 2 GBM with no class balancing, labels whose margin is measured against
    hybrid, and an always-hybrid fallback: epsilon is chosen by inner 3-fold CV and the router is used only
    if it beats always-hybrid on those inner folds;
  * primary test: pooled held-out nDCG@10(router) - nDCG@10(always-hybrid), paired bootstrap; supported iff
    the 95% CI lower bound > 0; non-inferior iff it is > -0.005;
  * secondary: blended 0.6 nDCG@10 + 0.4 MRR@10 target, + query-embedding features, no fallback, and the
    Experiment 2 router refit, each vs always-hybrid; per dataset; MRR@10 and MAP on the chosen route.

Usage:
  python scripts/run_experiment2d_cost_aware_router.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
from router_retrieval.labeling.margin_labeler import MarginLabeler  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp2c = _load_script("exp2c", "run_experiment2c_embedding_router.py")
exp2, exp1b = exp2c.exp2, exp2c.exp1b

log = get_logger("experiment2d")
PREREG = "results/preregistration/experiment2d_cost_aware_router_20261031.md"
METRICS = exp2c.METRICS
ROUTES = exp2.ROUTES
BLEND = {"ndcg_cut_10": 0.6, "mrr_10": 0.4}
NONINF_MARGIN = 0.005
HYBRID = "always_hybrid"  # the fallback candidate in the epsilon search
# name: (features, label/utility target, balanced classes, margin_to, hybrid fallback)
SYSTEMS = {
    "cost_aware": ("base", "ndcg", False, "default", True),
    "blended": ("base", "blend", False, "default", True),
    "embedding": ("emb", "ndcg", False, "default", True),
    "no_fallback": ("base", "ndcg", False, "default", False),
    "exp2_refit": ("base", "ndcg", True, "runner_up", False),
}
PRIMARY = "cost_aware"


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2d: cost-aware router vs tuned always-hybrid")
    p.add_argument("--datasets", nargs="+", default=exp2.DATASETS)
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--exp2_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--exp2c_dir", type=Path, default=Path("results/experiment2c"))
    p.add_argument("--output_dir", type=Path, default=Path("results/experiment2d"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------

def fit_router(X, R, eps, balanced, margin_to, router_cfg, seed):
    """Experiment 2's GBM router with the class weighting and label margin made explicit."""
    y = MarginLabeler(epsilon=eps, margin_to=margin_to).label_wide(R)["best_route"]
    weights = (R.max(axis=1) - R.min(axis=1) + 0.01).values  # regret spread, as in Experiment 2
    g = router_cfg["models"]["gbm"]
    model = exp2.GBMRouter(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"],
                           num_leaves=g["num_leaves"], class_weight="balanced" if balanced else None,
                           random_state=seed)
    if y.nunique() < 2:  # degenerate fold: constant router
        model.constant = y.iloc[0]
        return model
    model.constant = None
    return model.fit(X, y, sample_weight=weights)


def choose_epsilon(X, R, eps_grid, balanced, margin_to, fallback, router_cfg, seed):
    """Inner 3-fold CV over training queries; with `fallback`, always-hybrid is a candidate and wins ties."""
    utility = {}
    for eps in eps_grid:
        total = []
        for itr, ite in create_kfold_splits(list(X.index), exp2.INNER_FOLDS, seed):
            m = fit_router(X.loc[itr], R.loc[itr], eps, balanced, margin_to, router_cfg, seed)
            total.append(exp2.routed_scores(R.loc[ite], exp2.predict_routes(m, X.loc[ite])))
        utility[eps] = float(np.concatenate(total).mean())
    best = max(eps_grid, key=lambda e: (utility[e], -e))  # ties -> the smaller epsilon, as in Experiment 2
    if fallback and utility[best] <= float(R["hybrid"].mean()):
        return HYBRID, utility
    return best, utility


def fit_fold(df, emb, tables, train_ids, test_ids, alphas, eps_grid, router_cfg, seed):
    alpha = max(alphas, key=lambda a: df.loc[train_ids, f"ndcg_hyb_{a}"].mean())
    E_tr, E_te = exp2c.embedding_features(emb, train_ids, test_ids, seed)
    feats = {
        "base": (df.loc[train_ids, exp2.FEATURES], df.loc[test_ids, exp2.FEATURES]),
        "emb": (df.loc[train_ids, exp2.FEATURES].join(E_tr), df.loc[test_ids, exp2.FEATURES].join(E_te)),
    }
    cols = ["bm25", "dense", f"hyb_{alpha}"]
    R_m = {m: T[cols].set_axis(ROUTES, axis=1) for m, T in tables.items()}
    targets = {"ndcg": R_m["ndcg_cut_10"], "blend": sum(w * R_m[m] for m, w in BLEND.items())}

    info, routes = {"alpha": alpha}, {}
    for name, (feat, target, balanced, margin_to, fallback) in SYSTEMS.items():
        X_tr, X_te = feats[feat]
        R_tr = targets[target].loc[train_ids]
        eps, utility = choose_epsilon(X_tr, R_tr, eps_grid, balanced, margin_to, fallback, router_cfg, seed)
        info[name] = {"epsilon": eps, "inner_utility": utility, "inner_hybrid": float(R_tr["hybrid"].mean())}
        if eps == HYBRID:
            routes[name] = np.array(["hybrid"] * len(test_ids), dtype=object)
        else:
            m = fit_router(X_tr, R_tr, eps, balanced, margin_to, router_cfg, seed)
            routes[name] = exp2.predict_routes(m, X_te)
        info[name]["route_share"] = {r: float((routes[name] == r).mean()) for r in ROUTES}

    out = pd.DataFrame(index=test_ids)
    out["dataset"] = df.loc[test_ids, "dataset"].values
    props = np.array([(routes[PRIMARY] == r).mean() for r in ROUTES])
    for metric, R in R_m.items():
        R_te = R.loc[test_ids]
        out[f"hybrid_{metric}"] = R_te["hybrid"]
        out[f"rrf_{metric}"] = tables[metric].loc[test_ids, "rrf"]
        out[f"oracle_{metric}"] = R_te.max(axis=1)
        out[f"random_{metric}"] = R_te[ROUTES].to_numpy() @ props
        for name, r in routes.items():
            out[f"{name}_{metric}"] = exp2.routed_scores(R_te, r)
    for name, r in routes.items():
        out[f"route_{name}"] = r
    return out, info


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

# (PRIMARY, "rrf") was added after the pre-declaration, for H1 as worded in the proposal ("... and RRF").
COMPARISONS = [*((s, "hybrid") for s in SYSTEMS), (PRIMARY, "random"), (PRIMARY, "rrf"), ("oracle", "hybrid")]
fmt_ci = exp2c.fmt_ci


def summarize(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    systems = ["hybrid", "rrf", *SYSTEMS, "random", "oracle"]
    return {
        "n": len(df),
        "mean": {m: {s: float(df[f"{s}_{m}"].mean()) for s in systems} for m in METRICS},
        "tests": {m: {f"{a} - {b}": paired_bootstrap_test(df[f"{a}_{m}"].values, df[f"{b}_{m}"].values, n_boot, seed)
                      for a, b in COMPARISONS} for m in METRICS},
        "route_share": {s: {r: float((df[f"route_{s}"] == r).mean()) for r in ROUTES} for s in SYSTEMS},
    }


LABELS = {"cost_aware": "Cost-aware GBM", "blended": "Blended target", "embedding": "+ embedding",
          "no_fallback": "No fallback", "exp2_refit": "Exp 2 refit (balanced)"}


def write_tables(summary: dict, path: Path) -> None:
    prim = summary["pooled"]["tests"]["ndcg_cut_10"][f"{PRIMARY} - hybrid"]
    rows = [*summary["per_dataset"].items(), ("**pooled**", summary["pooled"])]
    lines = [
        "# Experiment 2d results: cost-aware router vs tuned always-hybrid (exploratory)",
        "",
        f"Design fixed before running in `{PREREG}` (not blind: a diagnostic on the same folds motivated it). "
        "Held-out queries, 5-fold CV within each dataset (Experiment 2 folds); α*, ε, PCA and routers fitted on "
        "training folds only.",
        "",
        f"**Primary (pooled, n = {summary['pooled']['n']}): cost-aware GBM − always-hybrid nDCG@10 = {fmt_ci(prim)}, "
        f"p = {prim['p_value']:.1e} → superiority {summary['primary_verdict'].upper()}; non-inferiority "
        f"(δ = {NONINF_MARGIN}) {summary['noninferiority_verdict'].upper()}.**",
        "",
    ]
    for m in METRICS:
        lines += [
            f"## {m}: means",
            "",
            "| Dataset | n | Always-hybrid | RRF | " + " | ".join(LABELS[s] for s in SYSTEMS) + " | Random (same mix) | Oracle |",
            "| :--- | :---: | " + " | ".join(":---:" for _ in range(len(SYSTEMS) + 4)) + " |",
        ]
        for name, s in rows:
            mm = s["mean"][m]
            lines.append(f"| {name} | {s['n']} | {mm['hybrid']:.4f} | {mm['rrf']:.4f} | "
                         + " | ".join(f"{mm[x]:.4f}" for x in SYSTEMS)
                         + f" | {mm['random']:.4f} | {mm['oracle']:.4f} |")
        lines += [
            "",
            f"## {m}: router − always-hybrid [95% bootstrap CI]",
            "",
            "| Dataset | " + " | ".join(LABELS[s] for s in SYSTEMS) + " | Cost-aware − random | Cost-aware − RRF (added) |",
            "| :--- | " + " | ".join(":---:" for _ in range(len(SYSTEMS) + 2)) + " |",
        ]
        for name, s in rows:
            t = s["tests"][m]
            lines.append(f"| {name} | " + " | ".join(fmt_ci(t[f"{x} - hybrid"]) for x in SYSTEMS)
                         + f" | {fmt_ci(t[f'{PRIMARY} - random'])} | {fmt_ci(t[f'{PRIMARY} - rrf'])} |")
        lines.append("")
    lines += [
        "## Route shares (held-out) and per-fold choices",
        "",
        "| Dataset | System | BM25 | Dense | Hybrid | ε per fold (`H` = always-hybrid fallback) |",
        "| :--- | :--- | :---: | :---: | :---: | :--- |",
    ]
    for ds, folds in summary["fold_info"].items():
        sh = summary["per_dataset"][ds]["route_share"]
        for s in SYSTEMS:
            eps = ", ".join("H" if f[s]["epsilon"] == HYBRID else "{:g}".format(f[s]["epsilon"]) for f in folds)
            lines.append(f"| {ds} | {LABELS[s]} | {sh[s]['bm25']:.1%} | {sh[s]['dense']:.1%} | {sh[s]['hybrid']:.1%} | {eps} |")
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
    emb_args = argparse.Namespace(**{**vars(args), "output_dir": args.exp2c_dir})  # reuse 2c's embedding cache
    out_dir = args.output_dir

    heldout, fold_info = [], {}
    for ds in args.datasets:
        df = pd.read_csv(args.exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        df.index = df.index.astype(str)
        path = args.processed_dir / "quora-sub" if ds == "quora" else args.data_dir / ds
        _, queries, qrels = exp1b.load_split(path)
        emb = exp2c.query_embeddings(ds, queries, emb_args).loc[df.index]
        tables = {m: t.loc[df.index] for m, t in
                  exp2c.route_metric_tables(ds, qrels, alphas, args, ret_cfg, hyb_cfg).items()}
        drift = (tables["ndcg_cut_10"][[f"hyb_{a}" for a in alphas]].to_numpy()
                 - df[[f"ndcg_hyb_{a}" for a in alphas]].to_numpy())
        if np.abs(drift).max() > 1e-6:
            raise RuntimeError(f"[{ds}] recomputed hybrid nDCG@10 differs from the Experiment 2 cache")

        fold_info[ds] = []
        for i, (tr, te) in enumerate(create_kfold_splits(list(df.index), k_folds, seed)):
            out, info = fit_fold(df, emb, tables, tr, te, alphas, eps_grid, router_cfg, seed)
            out["fold"] = i
            heldout.append(out)
            fold_info[ds].append(info)
        log.info(f"[{ds}] cost-aware ε per fold: " + ", ".join(str(f[PRIMARY]["epsilon"]) for f in fold_info[ds]))
    heldout = pd.concat(heldout)
    heldout.index = [f"{d}:{q}" for d, q in zip(heldout["dataset"], heldout.index)]

    summary = {
        "preregistration": PREREG,
        "status": "exploratory (not blind)",
        "pooled": summarize(heldout, args.n_bootstrap, seed),
        "per_dataset": {ds: summarize(heldout[heldout["dataset"] == ds], args.n_bootstrap, seed) for ds in fold_info},
        "fold_info": fold_info,
    }
    prim = summary["pooled"]["tests"]["ndcg_cut_10"][f"{PRIMARY} - hybrid"]
    summary["primary_verdict"] = "supported" if prim["ci_95_low"] > 0 else "not supported"
    summary["noninferiority_verdict"] = "met" if prim["ci_95_low"] > -NONINF_MARGIN else "not met"

    out_dir.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(out_dir / "per_query_heldout.csv")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    write_tables(summary, out_dir / "tables.md")
    log.info(f"PRIMARY: cost-aware GBM − always-hybrid (pooled n={len(heldout)}) = {fmt_ci(prim)} "
             f"p={prim['p_value']:.1e} → {summary['primary_verdict'].upper()}; "
             f"non-inferiority {summary['noninferiority_verdict'].upper()}")
    log.info(f"Wrote {out_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
