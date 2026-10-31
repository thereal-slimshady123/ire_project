"""Experiment 2c (exploratory, pre-declared): does adding the query embedding to the 18 features beat always-hybrid?

Design fixed in results/preregistration/experiment2c_embedding_router_20261031.md (hash in SHA256SUMS.txt):
  * Experiment 2's datasets, folds, labels, regret weights, alpha* and inner-CV epsilon, unchanged;
  * features = 18 Experiment 2 features + PCA-32 of the e5-small-v2 query embedding, PCA fitted on the
    outer training fold only;
  * primary test: pooled held-out nDCG@10(embedding GBM router) - nDCG@10(always-hybrid), paired
    bootstrap; supported iff the 95% CI lower bound > 0;
  * secondary: per dataset and for MRR@10 / MAP, vs always-hybrid and RRF (Holm over datasets x metrics),
    vs random routing with the same mix, vs the 18-feature GBM refit on the same folds, logistic variant.

Usage:
  python scripts/run_experiment2c_embedding_router.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.evaluation.ranking_metrics import compute_ranking_metrics  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
from router_retrieval.labeling.margin_labeler import MarginLabeler  # noqa: E402
from router_retrieval.retrieval.dense import DenseRetriever  # noqa: E402
from router_retrieval.retrieval.rrf import ReciprocalRankFusion  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp2 = _load_script("exp2", "run_experiment2_heldout_routing.py")
exp1, exp1b = exp2.exp1, exp2.exp1b

log = get_logger("experiment2c")
PREREG = "results/preregistration/experiment2c_embedding_router_20261031.md"
METRICS = ["ndcg_cut_10", "mrr_10", "map"]
ROUTES = exp2.ROUTES
N_PCA = 32
RRF_K = 60


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2c: query-embedding router vs tuned always-hybrid")
    p.add_argument("--datasets", nargs="+", default=exp2.DATASETS)
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--exp2_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--output_dir", type=Path, default=Path("results/experiment2c"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Inputs: query embeddings and per-route metric tables
# --------------------------------------------------------------------------------------

def query_embeddings(ds: str, queries, args) -> pd.DataFrame:
    """L2-normalized e5 query embeddings ("query: " prefix), cached per dataset."""
    cache = args.output_dir / "cache" / f"{ds}_query_emb.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    dense = DenseRetriever(model_name=args.dense_model, device=args.device, query_prefix="query: ")
    ids = sorted(queries)
    emb = dense.encode([dense.query_prefix + queries[q].text for q in ids])
    df = pd.DataFrame(emb, index=pd.Index(ids, name="query_id"), columns=[f"e{i}" for i in range(emb.shape[1])])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def route_metric_tables(ds: str, qrels, alphas, args, ret_cfg, hyb_cfg) -> Dict[str, pd.DataFrame]:
    """{metric: per-query table with columns bm25, dense, rrf, hyb_<alpha>} from the saved runs."""
    k = ret_cfg["dense"].get("top_k", 100)
    run_dir = args.results_dir / "runs" / ds
    bm25_run = exp1.read_trec_run(run_dir / f"bm25_k1-{ret_cfg['bm25']['k1']}_b-{ret_cfg['bm25']['b']}.trec")
    dense_run = exp1.read_trec_run(run_dir / f"dense_{args.dense_model.replace('/', '__')}.trec")
    for q in qrels:
        bm25_run.setdefault(q, [])
        dense_run.setdefault(q, [])
    rrf = ReciprocalRankFusion(k=RRF_K)
    runs = {"bm25": bm25_run, "dense": dense_run,
            "rrf": {q: rrf.fuse([bm25_run[q], dense_run[q]], top_k=k) for q in qrels}}
    hyb = exp1.build_hybrid_runs(bm25_run, dense_run, alphas, k, hyb_cfg["normalization"]["method"])
    runs.update({f"hyb_{a}": hyb[a] for a in alphas})
    per_run = {name: compute_ranking_metrics(qrels, run, METRICS) for name, run in runs.items()}
    return {m: pd.DataFrame({name: t[m] for name, t in per_run.items()}) for m in METRICS}


# --------------------------------------------------------------------------------------
# Fitting (Experiment 2's fit_router, with a fold-specific feature matrix)
# --------------------------------------------------------------------------------------

def embedding_features(emb: pd.DataFrame, train_ids, test_ids, seed: int):
    """PCA-32 of the query embedding, fitted on training queries only."""
    pca = PCA(n_components=N_PCA, random_state=seed).fit(emb.loc[train_ids].to_numpy())
    cols = [f"emb_pc{i}" for i in range(N_PCA)]
    return (pd.DataFrame(pca.transform(emb.loc[ids].to_numpy()), index=ids, columns=cols)
            for ids in (train_ids, test_ids))


def tune_epsilon(X: pd.DataFrame, R: pd.DataFrame, eps_grid, router_cfg, seed) -> float:
    """Experiment 2's inner 3-fold CV over training queries, on the given feature matrix."""
    def utility(eps):
        total = []
        for itr, ite in create_kfold_splits(list(X.index), exp2.INNER_FOLDS, seed):
            m = exp2.fit_router("gbm", X.loc[itr], R.loc[itr], eps, router_cfg, seed)
            total.append(exp2.routed_scores(R.loc[ite], exp2.predict_routes(m, X.loc[ite])))
        return float(np.concatenate(total).mean())

    return MarginLabeler().tune_epsilon(R, eps_grid, utility_fn=utility)


def fit_fold(df, emb, tables, train_ids, test_ids, alphas, eps_grid, router_cfg, seed):
    alpha = max(alphas, key=lambda a: df.loc[train_ids, f"ndcg_hyb_{a}"].mean())
    E_tr, E_te = embedding_features(emb, train_ids, test_ids, seed)
    feats = {
        "emb": (df.loc[train_ids, exp2.FEATURES].join(E_tr), df.loc[test_ids, exp2.FEATURES].join(E_te)),
        "base": (df.loc[train_ids, exp2.FEATURES], df.loc[test_ids, exp2.FEATURES]),
    }
    R_tr = exp2.route_matrix(df.loc[train_ids], alpha)
    info = {"alpha": alpha}
    routes = {}
    for name, (X_tr, X_te) in feats.items():
        eps = tune_epsilon(X_tr, R_tr, eps_grid, router_cfg, seed)
        info[f"epsilon_{name}"] = eps
        kinds = ("gbm", "logistic") if name == "emb" else ("gbm",)
        for kind in kinds:
            m = exp2.fit_router(kind, X_tr, R_tr, eps, router_cfg, seed)
            routes[f"{kind}_{name}"] = exp2.predict_routes(m, X_te)

    out = pd.DataFrame(index=test_ids)
    out["dataset"] = df.loc[test_ids, "dataset"].values
    for metric, T in tables.items():
        R_te = T.loc[test_ids, ["bm25", "dense", f"hyb_{alpha}"]].set_axis(ROUTES, axis=1)
        out[f"hybrid_{metric}"] = R_te["hybrid"]
        out[f"rrf_{metric}"] = T.loc[test_ids, "rrf"]
        for sys_name, r in routes.items():
            out[f"router_{sys_name}_{metric}"] = exp2.routed_scores(R_te, r)
        props = np.array([(routes["gbm_emb"] == r).mean() for r in ROUTES])
        out[f"random_gbm_emb_{metric}"] = R_te[ROUTES].to_numpy() @ props
    for sys_name, r in routes.items():
        out[f"route_{sys_name}"] = r
    info["route_share_gbm_emb"] = dict(zip(ROUTES, props.round(4).tolist()))
    return out, info


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

COMPARISONS = [
    ("router_gbm_emb", "hybrid"),
    ("router_gbm_emb", "rrf"),
    ("router_gbm_emb", "random_gbm_emb"),
    ("router_gbm_emb", "router_gbm_base"),
    ("router_logistic_emb", "hybrid"),
]


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    adj, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[key]))
        adj[key] = running
    return adj


def summarize(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    systems = ["hybrid", "rrf", "router_gbm_emb", "router_gbm_base", "router_logistic_emb", "random_gbm_emb"]
    return {
        "n": len(df),
        "mean": {m: {s: float(df[f"{s}_{m}"].mean()) for s in systems} for m in METRICS},
        "tests": {m: {f"{a} - {b}": paired_bootstrap_test(df[f"{a}_{m}"].values, df[f"{b}_{m}"].values, n_boot, seed)
                      for a, b in COMPARISONS} for m in METRICS},
    }


def fmt_ci(t: dict) -> str:
    return f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"


def write_tables(summary: dict, path: Path) -> None:
    prim = summary["pooled"]["tests"]["ndcg_cut_10"]["router_gbm_emb - hybrid"]
    rows = [*summary["per_dataset"].items(), ("**pooled**", summary["pooled"])]
    lines = [
        "# Experiment 2c results: query-embedding router vs tuned always-hybrid (exploratory)",
        "",
        f"Design fixed before running in `{PREREG}`. Held-out queries, 5-fold CV within each dataset (same folds "
        "as Experiment 2); α*, ε, PCA and routers fitted on training folds only. Routes are chosen with nDCG@10 "
        "labels; MRR@10 and MAP are measured on the chosen route.",
        "",
        f"**Primary (pooled, n = {summary['pooled']['n']}): embedding GBM router − always-hybrid nDCG@10 = "
        f"{fmt_ci(prim)}, p = {prim['p_value']:.1e} → {summary['primary_verdict'].upper()}.**",
        "",
    ]
    for m in METRICS:
        lines += [
            f"## {m}: means",
            "",
            "| Dataset | n | Always-hybrid | RRF | GBM (18 feat.) | GBM (18 + emb) | Logistic (18 + emb) | Random (same mix) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for name, s in rows:
            mm = s["mean"][m]
            lines.append(f"| {name} | {s['n']} | {mm['hybrid']:.4f} | {mm['rrf']:.4f} | {mm['router_gbm_base']:.4f} "
                         f"| {mm['router_gbm_emb']:.4f} | {mm['router_logistic_emb']:.4f} | {mm['random_gbm_emb']:.4f} |")
        lines += [
            "",
            f"## {m}: paired differences [95% bootstrap CI]",
            "",
            "| Dataset | GBM emb − hybrid (Holm p) | GBM emb − RRF | GBM emb − random | GBM emb − GBM 18 | Logistic emb − hybrid |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for name, s in rows:
            t = s["tests"][m]
            holm_p = summary["holm_router_vs_hybrid"].get(f"{name}/{m}")
            hp = f" (p={holm_p:.3f})" if holm_p is not None else ""
            lines.append(f"| {name} | {fmt_ci(t['router_gbm_emb - hybrid'])}{hp} | {fmt_ci(t['router_gbm_emb - rrf'])} "
                         f"| {fmt_ci(t['router_gbm_emb - random_gbm_emb'])} | {fmt_ci(t['router_gbm_emb - router_gbm_base'])} "
                         f"| {fmt_ci(t['router_logistic_emb - hybrid'])} |")
        lines.append("")
    lines += [
        "## Route shares of the embedding GBM router (held-out, mean over folds)",
        "",
        "| Dataset | BM25 | Dense | Hybrid | ε (emb) per fold | α* per fold |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, folds in summary["fold_info"].items():
        sh = {r: np.mean([f["route_share_gbm_emb"][r] for f in folds]) for r in ROUTES}
        eps_list = ", ".join("{:g}".format(f["epsilon_emb"]) for f in folds)
        alpha_list = ", ".join("{:g}".format(f["alpha"]) for f in folds)
        lines.append(f"| {name} | {sh['bm25']:.1%} | {sh['dense']:.1%} | {sh['hybrid']:.1%} | {eps_list} | {alpha_list} |")
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
        _, queries, qrels = exp1b.load_split(path)
        emb = query_embeddings(ds, queries, args).loc[df.index]
        tables = route_metric_tables(ds, qrels, alphas, args, ret_cfg, hyb_cfg)
        tables = {m: t.loc[df.index] for m, t in tables.items()}
        # The cached Experiment 2 nDCG@10 table must match the recomputed one (same runs, same fusion).
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
        log.info(f"[{ds}] folds done: " + "; ".join(f"α={f['alpha']:g} ε={f['epsilon_emb']:g}" for f in fold_info[ds]))
    heldout = pd.concat(heldout)
    heldout.index = [f"{d}:{q}" for d, q in zip(heldout["dataset"], heldout.index)]

    per_dataset = {ds: summarize(heldout[heldout["dataset"] == ds], args.n_bootstrap, seed) for ds in fold_info}
    summary = {
        "preregistration": PREREG,
        "status": "exploratory",
        "dense_model": args.dense_model,
        "n_pca": N_PCA,
        "pooled": summarize(heldout, args.n_bootstrap, seed),
        "per_dataset": per_dataset,
        "holm_router_vs_hybrid": holm({f"{ds}/{m}": s["tests"][m]["router_gbm_emb - hybrid"]["p_value"]
                                       for ds, s in per_dataset.items() for m in METRICS}),
        "fold_info": fold_info,
    }
    prim = summary["pooled"]["tests"]["ndcg_cut_10"]["router_gbm_emb - hybrid"]
    summary["primary_verdict"] = "supported" if prim["ci_95_low"] > 0 else "not supported"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(args.output_dir / "per_query_heldout.csv")
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    write_tables(summary, args.output_dir / "tables.md")
    log.info(f"PRIMARY: embedding GBM router − always-hybrid (pooled n={len(heldout)}) = {fmt_ci(prim)} "
             f"p={prim['p_value']:.1e} → {summary['primary_verdict'].upper()}")
    log.info(f"Wrote {args.output_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
