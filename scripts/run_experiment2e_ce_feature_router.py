"""Experiment 2e (exploratory, pre-declared): cost-aware router + document-aware cross-encoder features.

Design fixed in results/preregistration/experiment2e_ce_feature_router_20261031.md (hash in SHA256SUMS.txt):
  * Experiment 2's datasets, folds and training-fold alpha*;
  * 10 new features: cached MiniLM-L6 cross-encoder scores of the top-3 documents of BM25, dense and
    hybrid(alpha*): top-1 and top-3 mean per route, and BM25 - hybrid / dense - hybrid differences;
  * primary router = the Experiment 2d cost-aware GBM on the 18 + 10 features;
  * primary test: pooled held-out nDCG@10(router) - nDCG@10(always-hybrid), paired bootstrap; supported iff
    the 95% CI lower bound > 0; non-inferior iff it is > -0.005;
  * secondary: the same router on the 10 new features only, the Experiment 2b advantage router on 18 + 10,
    random routing at the same mix, and the Experiment 2d router; the cross-encoder cost per query.

Usage:
  python scripts/run_experiment2e_ce_feature_router.py
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
from router_retrieval.router.gbm_router import AdvantageRouter  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp2d = _load_script("exp2d", "run_experiment2d_cost_aware_router.py")
exp2c, exp2 = exp2d.exp2c, exp2d.exp2
exp1, exp1b = exp2c.exp1, exp2c.exp1b

log = get_logger("experiment2e")
PREREG = "results/preregistration/experiment2e_ce_feature_router_20261031.md"
METRICS = exp2c.METRICS
ROUTES = exp2.ROUTES
TOP_K = 3
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_POOL = 50  # pairs scored in one measured Experiment 3.5 rerank
NONINF_MARGIN = exp2d.NONINF_MARGIN
CE_FEATURES = [f"ce_{r}_{s}" for r in ROUTES for s in ("top1", "mean3")] + [
    f"ce_{r}_minus_hybrid_{s}" for r in ("bm25", "dense") for s in ("top1", "mean3")]
SYSTEMS = {"ce_router": "Cost-aware GBM, 18 + CE", "ce_only": "Cost-aware GBM, CE only",
           "ce_advantage": "Advantage router, 18 + CE", "exp2d": "Cost-aware GBM, 18 (Exp 2d)"}
PRIMARY = "ce_router"


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2e: router with cross-encoder relevance features")
    p.add_argument("--datasets", nargs="+", default=exp2.DATASETS)
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--exp2_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--output_dir", type=Path, default=Path("results/experiment2e"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Cross-encoder features
# --------------------------------------------------------------------------------------

def load_ce_cache(ds: str, results_dir: Path) -> dict:
    scores = {}
    path = results_dir / "runs" / ds / f"ce_{CE_MODEL.replace('/', '__')}.tsv"
    for line in open(path, encoding="utf-8"):
        q, d, s = line.rstrip("\n").split("\t")
        scores.setdefault(q, {})[d] = float(s)
    return scores


def ce_features(qids, runs: dict, ce: dict) -> pd.DataFrame:
    """CE features per query for one hybrid alpha; `runs` maps route -> run. Uncached pairs are NaN."""
    rows, pairs = {}, []
    for q in qids:
        row, docs = {}, set()
        for r in ROUTES:
            top = [c.doc_id for c in runs[r].get(q, [])[:TOP_K]]
            docs.update(top)
            s = [ce.get(q, {}).get(d, np.nan) for d in top]
            row[f"ce_{r}_top1"] = s[0] if s else np.nan
            row[f"ce_{r}_mean3"] = float(np.nanmean(s)) if s and not np.all(np.isnan(s)) else np.nan
        for r in ("bm25", "dense"):
            for stat in ("top1", "mean3"):
                row[f"ce_{r}_minus_hybrid_{stat}"] = row[f"ce_{r}_{stat}"] - row[f"ce_hybrid_{stat}"]
        rows[q] = row
        pairs.append(len(docs))
    return pd.DataFrame.from_dict(rows, orient="index")[CE_FEATURES], np.array(pairs)


def dataset_ce_features(ds: str, qids, alphas, args, ret_cfg, hyb_cfg):
    run_dir = args.results_dir / "runs" / ds
    bm25 = exp1.read_trec_run(run_dir / f"bm25_k1-{ret_cfg['bm25']['k1']}_b-{ret_cfg['bm25']['b']}.trec")
    dense = exp1.read_trec_run(run_dir / f"dense_{args.dense_model.replace('/', '__')}.trec")
    hyb = exp1.build_hybrid_runs(bm25, dense, alphas, ret_cfg["dense"].get("top_k", 100), hyb_cfg["normalization"]["method"])
    ce = load_ce_cache(ds, args.results_dir)
    return {a: ce_features(qids, {"bm25": bm25, "dense": dense, "hybrid": hyb[a]}, ce) for a in alphas}


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------

def cost_aware_routes(X_tr, X_te, R_tr, eps_grid, router_cfg, seed):
    eps, _ = exp2d.choose_epsilon(X_tr, R_tr, eps_grid, False, "default", True, router_cfg, seed)
    if eps == exp2d.HYBRID:
        return np.array(["hybrid"] * len(X_te), dtype=object), eps
    m = exp2d.fit_router(X_tr, R_tr, eps, False, "default", router_cfg, seed)
    return exp2.predict_routes(m, X_te), eps


def advantage_routes(X_tr, X_te, R_tr, adv_cfg, seed):
    keys = ("backend", "objective", "n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample",
            "colsample_bytree", "reg_lambda", "theta_grid", "label_epsilon", "inner_folds", "min_gain_z")
    m = AdvantageRouter(**{k: adv_cfg[k] for k in keys}, random_state=seed)
    m.fit(X_tr, R_tr["bm25"] - R_tr["hybrid"], R_tr["dense"] - R_tr["hybrid"])
    return np.array(m.predict(X_te), dtype=object)


def fit_fold(df, ce_by_alpha, tables, train_ids, test_ids, alphas, eps_grid, router_cfg, seed):
    alpha = max(alphas, key=lambda a: df.loc[train_ids, f"ndcg_hyb_{a}"].mean())
    C = ce_by_alpha[alpha][0]
    base = df[exp2.FEATURES]
    full = base.join(C)
    R_m = {m: T[["bm25", "dense", f"hyb_{alpha}"]].set_axis(ROUTES, axis=1) for m, T in tables.items()}
    R_tr = R_m["ndcg_cut_10"].loc[train_ids]

    routes, info = {}, {"alpha": alpha}
    routes["ce_router"], info["eps_ce_router"] = cost_aware_routes(
        full.loc[train_ids], full.loc[test_ids], R_tr, eps_grid, router_cfg, seed)
    routes["ce_only"], info["eps_ce_only"] = cost_aware_routes(
        C.loc[train_ids], C.loc[test_ids], R_tr, eps_grid, router_cfg, seed)
    routes["ce_advantage"] = advantage_routes(full.loc[train_ids], full.loc[test_ids], R_tr, router_cfg["advantage"], seed)
    routes["exp2d"], info["eps_exp2d"] = cost_aware_routes(
        base.loc[train_ids], base.loc[test_ids], R_tr, eps_grid, router_cfg, seed)
    info["route_share"] = {s: {r: float((routes[s] == r).mean()) for r in ROUTES} for s in routes}

    out = pd.DataFrame(index=test_ids)
    out["dataset"] = df.loc[test_ids, "dataset"].values
    out["ce_pairs"] = pd.Series(ce_by_alpha[alpha][1], index=C.index).loc[test_ids].values
    props = np.array([(routes[PRIMARY] == r).mean() for r in ROUTES])
    for metric, R in R_m.items():
        R_te = R.loc[test_ids]
        out[f"hybrid_{metric}"] = R_te["hybrid"]
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

COMPARISONS = [*((s, "hybrid") for s in SYSTEMS), (PRIMARY, "random"), (PRIMARY, "exp2d"), ("oracle", "hybrid")]
fmt_ci = exp2c.fmt_ci


def summarize(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    systems = ["hybrid", *SYSTEMS, "random", "oracle"]
    return {
        "n": len(df),
        "mean": {m: {s: float(df[f"{s}_{m}"].mean()) for s in systems} for m in METRICS},
        "tests": {m: {f"{a} - {b}": paired_bootstrap_test(df[f"{a}_{m}"].values, df[f"{b}_{m}"].values, n_boot, seed)
                      for a, b in COMPARISONS} for m in METRICS},
        "route_share": {s: {r: float((df[f"route_{s}"] == r).mean()) for r in ROUTES} for s in SYSTEMS},
        "mean_ce_pairs": float(df["ce_pairs"].mean()),
    }


def ce_cost_per_pair(results_dir: Path) -> dict:
    """Measured mean cross-encoder rerank time / pairs, per dataset with measured components."""
    path = results_dir / "policy_latency" / "summary.json"
    comps = json.loads(path.read_text(encoding="utf-8"))["components_ms"]
    return {ds: c["cross_encoder_rerank"]["mean"] / CE_POOL for ds, c in comps.items()}


def write_tables(summary: dict, path: Path) -> None:
    prim = summary["pooled"]["tests"]["ndcg_cut_10"][f"{PRIMARY} - hybrid"]
    rows = [*summary["per_dataset"].items(), ("**pooled**", summary["pooled"])]
    lines = [
        "# Experiment 2e results: router with cross-encoder relevance features (exploratory)",
        "",
        f"Design fixed before running in `{PREREG}` (not blind). Held-out queries, 5-fold CV within each dataset "
        "(Experiment 2 folds); α*, ε and routers fitted on training folds only.",
        "",
        f"**Primary (pooled, n = {summary['pooled']['n']}): cost-aware GBM with CE features − always-hybrid nDCG@10 = "
        f"{fmt_ci(prim)}, p = {prim['p_value']:.1e} → superiority {summary['primary_verdict'].upper()}; "
        f"non-inferiority (δ = {NONINF_MARGIN}) {summary['noninferiority_verdict'].upper()}.**",
        "",
    ]
    for m in METRICS:
        lines += [f"## {m}: router − always-hybrid [95% bootstrap CI]", "",
                  "| Dataset | n | Always-hybrid | " + " | ".join(SYSTEMS.values()) + " | Primary − random | Primary − Exp 2d |",
                  "| :--- | :---: | :---: | " + " | ".join(":---:" for _ in range(len(SYSTEMS) + 2)) + " |"]
        for name, s in rows:
            t = s["tests"][m]
            lines.append(f"| {name} | {s['n']} | {s['mean'][m]['hybrid']:.4f} | "
                         + " | ".join(fmt_ci(t[f"{x} - hybrid"]) for x in SYSTEMS)
                         + f" | {fmt_ci(t[f'{PRIMARY} - random'])} | {fmt_ci(t[f'{PRIMARY} - exp2d'])} |")
        lines.append("")
    lines += ["## Route shares (held-out)", "",
              "| Dataset | " + " | ".join(f"{v} (B/D/H)" for v in SYSTEMS.values()) + " |",
              "| :--- | " + " | ".join(":---:" for _ in SYSTEMS) + " |"]
    for name, s in rows:
        lines.append(f"| {name} | " + " | ".join(
            "/".join(f"{s['route_share'][x][r]:.0%}" for r in ROUTES) for x in SYSTEMS) + " |")
    cost = summary["ce_cost"]
    lines += ["", "## Cost of the CE features", "",
              f"Mean cross-encoder pairs per query (union of the three top-{TOP_K} lists): "
              f"{summary['pooled']['mean_ce_pairs']:.2f}. Measured CE time per pair (Experiment 3.5, 50-pair rerank / 50): "
              + ", ".join(f"{ds} {v:.2f} ms" for ds, v in cost["ms_per_pair"].items())
              + f". Estimated added latency per query: {cost['added_ms_mean']:.1f} ms (mean over the measured datasets)."]
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

    heldout, fold_info = [], {}
    for ds in args.datasets:
        df = pd.read_csv(args.exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        df.index = df.index.astype(str)
        path = args.processed_dir / "quora-sub" if ds == "quora" else args.data_dir / ds
        _, _, qrels = exp1b.load_split(path)
        tables = {m: t.loc[df.index] for m, t in
                  exp2c.route_metric_tables(ds, qrels, alphas, args, ret_cfg, hyb_cfg).items()}
        folds = list(create_kfold_splits(list(df.index), k_folds, seed))
        used = sorted({max(alphas, key=lambda a: df.loc[tr, f"ndcg_hyb_{a}"].mean()) for tr, _ in folds})
        ce_by_alpha = dataset_ce_features(ds, list(df.index), used, args, ret_cfg, hyb_cfg)
        log.info(f"[{ds}] CE features: {int(ce_by_alpha[used[0]][0].isna().any(axis=1).sum())} queries with a missing pair")

        fold_info[ds] = []
        for i, (tr, te) in enumerate(folds):
            out, info = fit_fold(df, ce_by_alpha, tables, tr, te, alphas, eps_grid, router_cfg, seed)
            out["fold"] = i
            heldout.append(out)
            fold_info[ds].append(info)
        log.info(f"[{ds}] primary ε per fold: " + ", ".join(str(f["eps_ce_router"]) for f in fold_info[ds]))
    heldout = pd.concat(heldout)
    heldout.index = [f"{d}:{q}" for d, q in zip(heldout["dataset"], heldout.index)]

    per_pair = ce_cost_per_pair(args.results_dir)
    measured = heldout[heldout["dataset"].isin(per_pair)]
    added = float(np.mean([measured.loc[measured["dataset"] == ds, "ce_pairs"].mean() * v for ds, v in per_pair.items()]))
    summary = {
        "preregistration": PREREG,
        "status": "exploratory (not blind)",
        "pooled": summarize(heldout, args.n_bootstrap, seed),
        "per_dataset": {ds: summarize(heldout[heldout["dataset"] == ds], args.n_bootstrap, seed) for ds in fold_info},
        "fold_info": fold_info,
        "ce_cost": {"ms_per_pair": per_pair, "added_ms_mean": added},
    }
    prim = summary["pooled"]["tests"]["ndcg_cut_10"][f"{PRIMARY} - hybrid"]
    summary["primary_verdict"] = "supported" if prim["ci_95_low"] > 0 else "not supported"
    summary["noninferiority_verdict"] = "met" if prim["ci_95_low"] > -NONINF_MARGIN else "not met"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(args.output_dir / "per_query_heldout.csv")
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    write_tables(summary, args.output_dir / "tables.md")
    log.info(f"PRIMARY: CE-feature router − always-hybrid (pooled n={len(heldout)}) = {fmt_ci(prim)} "
             f"p={prim['p_value']:.1e} → {summary['primary_verdict'].upper()}; "
             f"non-inferiority {summary['noninferiority_verdict'].upper()}")
    log.info(f"Wrote {args.output_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
