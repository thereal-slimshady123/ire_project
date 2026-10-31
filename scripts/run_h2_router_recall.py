"""H2 for the routing system: does held-out Recall@100 of the routed list match always-hybrid?

Reranking alone cannot change Recall@100 (it reorders a fixed pool), so the falsifiable part of H2 is routing:
a query sent to BM25-only or dense-only gets a different candidate set. No model is refitted; this scores the
saved held-out decisions:
  * Experiment 2 GBM and logistic routers (pre-registered), with each fold's alpha*;
  * Experiment 2c embedding GBM router (exploratory);
  * Experiment 2d cost-aware GBM router (exploratory; same folds and alpha* as Experiment 2);
  * router + tau escalation at each target rate (an escalated query gets the reranked hybrid top-50 + tail,
    whose Recall@100 equals hybrid's).

Decision rule, fixed before computing: with the 95% paired bootstrap CI of router - hybrid Recall@100,
  supported      if the CI lies within [-0.01, +0.01] (equivalence) or entirely above 0;
  not supported  if the CI lies entirely below 0 and its lower end is below -0.01;
  inconclusive   otherwise.

Usage:
  python scripts/run_h2_router_recall.py
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

from router_retrieval.evaluation.ranking_metrics import compute_ranking_metrics  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp2 = _load_script("exp2", "run_experiment2_heldout_routing.py")
exp1, exp1b = exp2.exp1, exp2.exp1b

log = get_logger("h2_router_recall")
METRIC = "recall_100"
MARGIN = 0.01


def parse_args():
    p = argparse.ArgumentParser(description="H2: held-out Recall@100 of routed lists vs always-hybrid")
    p.add_argument("--datasets", nargs="+", default=exp2.DATASETS)
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output_dir", type=Path, default=Path("results/h2_router_recall"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    return p.parse_args()


def verdict(t: dict) -> str:
    lo, hi = t["ci_95_low"], t["ci_95_high"]
    if (lo >= -MARGIN and hi <= MARGIN) or lo > 0:
        return "supported"
    if hi < 0 and lo < -MARGIN:
        return "not supported"
    return "inconclusive"


def fmt_ci(t: dict) -> str:
    return f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"


def main():
    args = parse_args()
    ret_cfg = load_config(args.config_dir / "retrievers.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")
    seed = load_config(args.config_dir / "experiment.yaml")["seed"]
    k = ret_cfg["dense"].get("top_k", 100)
    rates = [float(r) for r in load_config(args.config_dir / "escalation.yaml")["target_escalation_rates"]]

    e2 = pd.read_csv(args.results_dir / "experiment2" / "per_query_heldout.csv", index_col=0)
    e2c = pd.read_csv(args.results_dir / "experiment2c" / "per_query_heldout.csv", index_col=0)
    tau = pd.read_csv(args.results_dir / "tau_escalation" / "per_query_heldout.csv", index_col=0)
    e2d = pd.read_csv(args.results_dir / "experiment2d" / "per_query_heldout.csv", index_col=0)
    folds = json.loads((args.results_dir / "experiment2" / "summary.json").read_text(encoding="utf-8"))["fold_info"]

    rows = []
    for ds in args.datasets:
        path = args.processed_dir / "quora-sub" if ds == "quora" else args.data_dir / ds
        _, queries, qrels = exp1b.load_split(path)
        run_dir = args.results_dir / "runs" / ds
        bm25_run = exp1.read_trec_run(run_dir / f"bm25_k1-{ret_cfg['bm25']['k1']}_b-{ret_cfg['bm25']['b']}.trec")
        dense_run = exp1.read_trec_run(run_dir / f"dense_{args.dense_model.replace('/', '__')}.trec")
        for q in queries:
            bm25_run.setdefault(q, [])
            dense_run.setdefault(q, [])
        used = sorted({f["alpha"] for f in folds[ds]})
        hyb = exp1.build_hybrid_runs(bm25_run, dense_run, used, k, hyb_cfg["normalization"]["method"])
        rec = {"bm25": compute_ranking_metrics(qrels, bm25_run, [METRIC])[METRIC],
               "dense": compute_ranking_metrics(qrels, dense_run, [METRIC])[METRIC]}
        rec.update({a: compute_ranking_metrics(qrels, hyb[a], [METRIC])[METRIC] for a in used})

        sub = e2[e2["dataset"] == ds]
        qids = [i.split(":", 1)[1] for i in sub.index]
        alpha = [folds[ds][f]["alpha"] for f in sub["fold"]]
        hybrid = np.array([rec[a][q] for a, q in zip(alpha, qids)])

        def routed(routes):
            return np.array([rec[a][q] if r == "hybrid" else rec[r][q] for r, a, q in zip(routes, alpha, qids)])

        frame = pd.DataFrame(index=sub.index)
        frame["dataset"] = ds
        frame["hybrid"] = hybrid
        frame["router_gbm"] = routed(sub["route_gbm"])
        frame["router_logistic"] = routed(sub["route_logistic"])
        frame["router_gbm_emb"] = routed(e2c.loc[sub.index, "route_gbm_emb"])
        frame["router_cost_aware"] = routed(e2d.loc[sub.index, "route_cost_aware"])
        t = tau.loc[sub.index]
        tau_routed = routed(t["route"])
        for r in rates:
            frame[f"router_tau@{r}"] = np.where(t[f"escalated@{r}"].astype(bool), hybrid, tau_routed)
        rows.append(frame)
    df = pd.concat(rows)

    systems = [("router_gbm", "GBM router (Exp 2, pre-registered)"),
               ("router_logistic", "Logistic router (Exp 2, pre-registered)"),
               ("router_gbm_emb", "Embedding GBM router (Exp 2c, exploratory)"),
               ("router_cost_aware", "Cost-aware GBM router (Exp 2d, exploratory)"),
               *[(f"router_tau@{r}", f"GBM router + τ escalation, target {r:.0%} (exploratory)") for r in rates]]
    groups = [*[(ds, df[df["dataset"] == ds]) for ds in args.datasets], ("pooled", df)]
    summary = {"decision_rule": f"supported iff CI within ±{MARGIN} or entirely > 0; not supported iff CI < 0 "
                                f"and below -{MARGIN}; else inconclusive", "results": {}}
    for name, g in groups:
        summary["results"][name] = {"n": len(g), "mean_hybrid": float(g["hybrid"].mean()), "systems": {}}
        for col, _ in systems:
            test = paired_bootstrap_test(g[col].values, g["hybrid"].values, args.n_bootstrap, seed)
            summary["results"][name]["systems"][col] = {"mean": float(g[col].mean()), **test, "verdict": verdict(test)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "per_query_heldout.csv")
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    pooled = summary["results"]["pooled"]
    lines = ["# H2: held-out Recall@100 of the routing system vs always-hybrid", "",
             "Saved held-out route decisions (no refitting); hybrid at each training fold's α*. Escalated queries keep "
             "hybrid's candidate set. Decision rule fixed before computing: " + summary["decision_rule"] + ".", "",
             f"## Pooled (n = {pooled['n']}, always-hybrid Recall@100 = {pooled['mean_hybrid']:.4f})", "",
             "| System | Recall@100 | − hybrid [95% CI] | Verdict |", "| :--- | :---: | :---: | :---: |"]
    for col, label in systems:
        s = pooled["systems"][col]
        lines.append(f"| {label} | {s['mean']:.4f} | {fmt_ci(s)} | {s['verdict']} |")
    lines += ["", "## Per dataset: system − hybrid [95% CI]", "",
              "| Dataset | n | Hybrid | GBM (Exp 2) | Embedding GBM (Exp 2c) | Cost-aware GBM (Exp 2d) | GBM + τ, 10% |",
              "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for name, _ in groups:
        r = summary["results"][name]
        s = r["systems"]
        lines.append(f"| {name} | {r['n']} | {r['mean_hybrid']:.4f} | {fmt_ci(s['router_gbm'])} "
                     f"| {fmt_ci(s['router_gbm_emb'])} | {fmt_ci(s['router_cost_aware'])} | {fmt_ci(s['router_tau@0.1'])} |")
    (args.output_dir / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for col, label in systems:
        s = pooled["systems"][col]
        log.info(f"{label}: {fmt_ci(s)} → {s['verdict'].upper()}")


if __name__ == "__main__":
    main()
