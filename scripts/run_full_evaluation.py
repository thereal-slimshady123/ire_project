"""Combine completed held-out experiments into a single comparison report.

The script consumes saved Experiment 2–4 artifacts. It marks systems unavailable
when their owning track has not supplied evaluation artifacts; it never fabricates
missing RRF, advantage-router, or latency results.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output_dir", type=Path, default=Path("results/metrics"))
    args = p.parse_args()
    root = args.results_dir
    tables = {}
    summaries = {i: _json(root / f"experiment{i}" / "summary.json") for i in (2, 3, 4)}
    held = {i: (pd.read_csv(root / f"experiment{i}" / "per_query_heldout.csv")
                if (root / f"experiment{i}" / "per_query_heldout.csv").exists() else None)
            for i in (2, 3, 4)}
    datasets = sorted(set().union(*(set(s.get("per_dataset", {})) for s in summaries.values() if s)))
    for ds in datasets:
        systems = {}
        if summaries[2] and ds in summaries[2]["per_dataset"]:
            s = summaries[2]["per_dataset"][ds]
            for name, col in (("always_hybrid", "hybrid"), ("best_single", "best_single"),
                              ("gbm_router", "router_gbm"), ("logistic_router", "router_logistic"),
                              ("random_route", "random_gbm"), ("oracle_route", "oracle")):
                systems[name] = {"ndcg_10_mean": s["mean"][col], "n": s["n"]}
        if summaries[3] and ds in summaries[3]["per_dataset"]:
            s = summaries[3]["per_dataset"][ds]
            for name, col in (("skip_dense_gbm", "router"), ("skip_dense_random", "random"),
                              ("always_bm25", "bm25"), ("skip_dense_oracle", "oracle"),
                              ("skip_dense_top_ratio", "H1_bm25_top_ratio")):
                systems[name] = {"ndcg_10_mean": s["mean"][col], "n": s["n"]}
        if summaries[4] and ds in summaries[4]["per_dataset"]:
            s = summaries[4]["per_dataset"][ds]
            systems["always_rerank_minilm_l6"] = {"ndcg_10_mean": s["always_rerank"], "n": s["n"]}
            for name, col in (("predicted_gain_escalation_10pct", "predicted_gain"),
                              ("confidence_escalation_10pct", "router_confidence"),
                              ("random_escalation_10pct", "random"), ("oracle_escalation_10pct", "oracle")):
                systems[name] = {"ndcg_10_mean": s["budgets"]["0.1"][col]["mean"], "n": s["n"]}
        out = {"dataset": ds, "primary_metric": "ndcg_cut_10", "systems": systems,
               "unavailable_systems": ["rrf", "advantage_regression_router", "latency"],
               "source": "saved Experiment 2–4 held-out summaries"}
        folder = args.output_dir / ds
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        tables[ds] = systems
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Combined held-out evaluation", "", "Primary metric: mean nDCG@10.",
             "Values are copied from saved held-out Experiment 2–4 summaries. Systems without completed input artifacts are listed as unavailable.",
             "", "| Dataset | System | nDCG@10 | Queries |", "| :--- | :--- | ---: | ---: |"]
    for ds, systems in tables.items():
        for name, val in systems.items():
            lines.append(f"| {ds} | {name} | {val['ndcg_10_mean']:.4f} | {val['n']} |")
    lines += ["", "Unavailable from current artifacts: RRF, advantage regression router, and component latency."]
    (args.output_dir / "combined.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
