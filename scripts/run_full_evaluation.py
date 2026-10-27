"""Combine completed held-out experiments into a single comparison report.

The script consumes saved Experiment 2–4 artifacts. It marks systems unavailable
when their owning track has not supplied evaluation artifacts; it never fabricates
missing RRF, advantage-router, or latency results.
"""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.evaluation.significance import paired_bootstrap_test, paired_wilcoxon_test
from router_retrieval.evaluation.router_metrics import (
    compute_router_classification_metrics,
    compute_win_loss_tie_counts,
)
from router_retrieval.labeling.margin_labeler import MarginLabeler


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _paired_comparison(frame, score_a, score_b, seed=42, n_bootstrap=10000):
    """Summarize a paired held-out comparison from per-query nDCG values."""
    if frame is None or score_a not in frame or score_b not in frame:
        return None
    pairs = frame[[score_a, score_b]].dropna()
    if pairs.empty:
        return None
    boot = paired_bootstrap_test(pairs[score_a], pairs[score_b], n_bootstrap, seed)
    wilcoxon = paired_wilcoxon_test(pairs[score_a], pairs[score_b])
    return {"system_a": score_a, "system_b": score_b, "n": len(pairs),
            "paired_bootstrap": boot, "paired_wilcoxon": wilcoxon,
            "win_loss_tie": compute_win_loss_tie_counts(pairs[score_a], pairs[score_b])}


# System name -> (artifact, per-query column) for systems whose held-out per-query nDCG@10 is saved.
PER_QUERY_SOURCES = {
    "always_hybrid": (2, "hybrid"), "best_single": (2, "best_single"), "gbm_router": (2, "router_gbm"),
    "logistic_router": (2, "router_logistic"), "random_route": (2, "random_gbm"), "oracle_route": (2, "oracle"),
    "skip_dense_gbm": (3, "router"), "skip_dense_random": (3, "random"), "skip_dense_oracle": (3, "oracle"),
    "skip_dense_top_ratio": (3, "H1_bm25_top_ratio"),
    "always_rerank_minilm_l6": (4, "reranked"), "predicted_gain_escalation_10pct": (4, "predicted_gain@0.1"),
    "confidence_escalation_10pct": (4, "router_confidence@0.1"), "random_escalation_10pct": (4, "random@0.1"),
    "oracle_escalation_10pct": (4, "oracle@0.1"),
    "router_tau_escalation_10pct": ("tau", "router_tau@0.1"), "router_random_escalation_10pct": ("tau", "router_random@0.1"),
    "hybrid_tau_escalation_10pct": ("tau", "hybrid_tau@0.1"),
}
# Contract 5 system name -> system name in results/policy_latency/summary.json.
LATENCY_SOURCES = {
    "always_bm25": "always_bm25", "always_hybrid": "always_hybrid", "gbm_router": "gbm_router",
    "always_rerank_minilm_l6": "always_rerank_minilm_l6", "skip_dense_gbm": "skip_dense_gbm",
    "predicted_gain_escalation_10pct": "predicted_gain_escalation_10%",
    "confidence_escalation_10pct": "router_confidence_escalation_10%",
    "random_escalation_10pct": "random_escalation_10%", "oracle_escalation_10pct": "oracle_escalation_10%",
    "router_tau_escalation_10pct": "router_tau_escalation_10%",
}


def _mean_ci(values, seed=42, n_bootstrap=10000):
    """Mean with a 95% bootstrap CI over queries."""
    v = pd.Series(values).dropna().to_numpy(dtype=float)
    boot = paired_bootstrap_test(v, [0.0] * len(v), n_bootstrap, seed)
    return {"mean": boot["delta_mean"], "ci_95": [boot["ci_95_low"], boot["ci_95_high"]]}


def _router_confusion(root, dataset, held_frame, summary):
    """Compare held-out router decisions with fold-specific margin labels (diagnostic only)."""
    cache_path = root / "experiment2" / "cache" / f"{dataset}.csv"
    fold_info = summary.get("fold_info", {}).get(dataset)
    if not cache_path.exists() or not fold_info or held_frame is None:
        return None
    cache = pd.read_csv(cache_path, index_col=0, dtype={"query_id": str})
    cache.index = cache.index.astype(str)
    rows = held_frame[held_frame["dataset"] == dataset].copy()
    rows["cache_id"] = rows["query_id"].astype(str).str.split(":", n=1).str[-1]
    labels = {"gbm": [], "logistic": []}
    predictions = {"gbm": [], "logistic": []}
    route_names = ["bm25", "dense", "hybrid"]
    for fold, info in enumerate(fold_info):
        part = rows[rows["fold"] == fold]
        ids = [qid for qid in part["cache_id"] if qid in cache.index]
        if not ids:
            continue
        scores = cache.loc[ids, ["ndcg_bm25", "ndcg_dense", f"ndcg_hyb_{info['alpha']:.1f}"]].copy()
        scores.columns = route_names
        truth = MarginLabeler(epsilon=float(info["epsilon"])).label_wide(scores)["best_route"].tolist()
        truth_by_id = dict(zip(ids, truth))
        indexed = part.set_index("cache_id")
        labels["gbm"].extend(truth_by_id[q] for q in ids)
        labels["logistic"].extend(truth_by_id[q] for q in ids)
        predictions["gbm"].extend(indexed.loc[ids, "route_gbm"].astype(str).tolist())
        predictions["logistic"].extend(indexed.loc[ids, "route_logistic"].astype(str).tolist())
    return {key: compute_router_classification_metrics(labels[key], predictions[key], route_names)
            for key in labels} if labels["gbm"] else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output_dir", type=Path, default=Path("results/metrics"))
    p.add_argument("--strong_experiment4_dir", type=Path, default=None,
                   help="Optional separate Experiment 4 directory for a stronger reranker run")
    p.add_argument("--reranker_benchmark_summary", type=Path, default=None,
                   help="Optional matched-pool cross-encoder benchmark summary.json")
    args = p.parse_args()
    root = args.results_dir
    tables = {}
    summaries = {i: _json(root / f"experiment{i}" / "summary.json") for i in (2, 3, 4)}
    held = {i: (pd.read_csv(root / f"experiment{i}" / "per_query_heldout.csv")
                if (root / f"experiment{i}" / "per_query_heldout.csv").exists() else None)
            for i in (2, 3, 4)}
    tau_summary = _json(root / "tau_escalation" / "summary.json")
    held["tau"] = (pd.read_csv(root / "tau_escalation" / "per_query_heldout.csv")
                   if (root / "tau_escalation" / "per_query_heldout.csv").exists() else None)
    latency = _json(root / "policy_latency" / "summary.json")
    latency_by = {(s["dataset"], s["system"]): s for s in (latency or {}).get("systems", [])}
    headroom = _json(root / "metrics" / "oracle_headroom.json")
    strong_summary = _json(args.strong_experiment4_dir / "summary.json") if args.strong_experiment4_dir else None
    reranker_benchmark = _json(args.reranker_benchmark_summary) if args.reranker_benchmark_summary else None
    strong_held = (pd.read_csv(args.strong_experiment4_dir / "per_query_heldout.csv")
                   if args.strong_experiment4_dir and
                   (args.strong_experiment4_dir / "per_query_heldout.csv").exists() else None)
    for frame in held.values():
        if frame is not None and "Unnamed: 0" in frame:
            frame.rename(columns={"Unnamed: 0": "query_id"}, inplace=True)
    datasets = sorted(set().union(*(set(s.get("per_dataset", {})) for s in summaries.values() if s)))
    for ds in datasets:
        systems = {}
        baseline = _json(root / "metrics" / ds / "baselines.json")
        if baseline:
            for name, source in (("always_bm25", baseline["bm25"]),
                                 ("always_dense", baseline["dense"]),
                                 ("always_hybrid", baseline["hybrid_cv"]),
                                 ("rrf", baseline["rrf"])):
                systems[name] = {"ndcg_10_mean": source["ndcg_cut_10"],
                                 "n": baseline["n_queries"],
                                 "metrics": {"ndcg_10": source["ndcg_cut_10"],
                                             "mrr_10": source["mrr_10"],
                                             "map": source["map"],
                                             "recall_100": source["recall_100"]}}
        if summaries[2] and ds in summaries[2]["per_dataset"]:
            s = summaries[2]["per_dataset"][ds]
            for name, col in (("always_hybrid", "hybrid"), ("best_single", "best_single"),
                              ("gbm_router", "router_gbm"), ("logistic_router", "router_logistic"),
                              ("random_route", "random_gbm"), ("oracle_route", "oracle")):
                systems.setdefault(name, {}).update({"ndcg_10_mean": s["mean"][col], "n": s["n"]})
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
        if tau_summary and ds in tau_summary["per_dataset"]:
            s = tau_summary["per_dataset"][ds]
            b = s["rates"]["0.1"]
            for name, col in (("router_tau_escalation_10pct", "router_tau"),
                              ("router_random_escalation_10pct", "router_random"),
                              ("hybrid_tau_escalation_10pct", "hybrid_tau")):
                systems[name] = {"ndcg_10_mean": b["mean"][col], "n": s["n"],
                                 "realized_escalation_rate": b["heldout_rate"]}
        if strong_summary and ds in strong_summary.get("per_dataset", {}):
            s = strong_summary["per_dataset"][ds]
            systems["always_rerank_stronger"] = {"ndcg_10_mean": s["always_rerank"], "n": s["n"]}
            for name, col in (("predicted_gain_escalation_stronger_10pct", "predicted_gain"),
                              ("confidence_escalation_stronger_10pct", "router_confidence"),
                              ("random_escalation_stronger_10pct", "random"),
                              ("oracle_escalation_stronger_10pct", "oracle")):
                systems[name] = {"ndcg_10_mean": s["budgets"]["0.1"][col]["mean"], "n": s["n"]}
        comparisons = {}
        benchmark_ds = (reranker_benchmark or {}).get("results", {}).get(ds)
        if benchmark_ds:
            for model, result in benchmark_ds.get("models", {}).items():
                label = f"reranker_{model.split('/')[-1].lower().replace('-', '_')}"
                systems[label] = {"ndcg_10_mean": result["mean_metrics"]["ndcg_cut_10"],
                                  "n": result["n"], "metrics": {
                                      "ndcg_10": result["mean_metrics"]["ndcg_cut_10"],
                                      "mrr_10": result["mean_metrics"]["mrr_10"],
                                      "map": result["mean_metrics"]["map"],
                                      "recall_100": result["mean_metrics"]["recall_100"]},
                                  "latency_ms": {"mean": result["rerank_latency_ms_mean"],
                                                 "p50": result["rerank_latency_ms_p50"],
                                                 "p95": result["rerank_latency_ms_p95"]}}
                delta = result["reranker_minus_hybrid"]
                comparisons[f"{label}_vs_hybrid"] = {
                    "n": delta["n"], "paired_bootstrap": {
                        "delta_mean": delta["delta_mean"], "ci_95_low": delta["ci_95_low"],
                        "ci_95_high": delta["ci_95_high"], "p_value": delta["p_value"]},
                    "paired_wilcoxon": delta.get("paired_wilcoxon"),
                    "win_loss_tie": delta.get("win_loss_tie")}
        for name, (exp_id, col) in PER_QUERY_SOURCES.items():
            frame = held[exp_id]
            if name in systems and frame is not None and col in frame and (frame["dataset"] == ds).any():
                systems[name]["ndcg_10"] = _mean_ci(frame.loc[frame["dataset"] == ds, col])
        for name, lat_name in LATENCY_SOURCES.items():
            if name in systems and (ds, lat_name) in latency_by:
                lat = latency_by[(ds, lat_name)]
                systems[name]["latency_ms"] = {"mean": lat["mean_ms"], "p50": lat["p50_ms"], "p95": lat["p95_ms"],
                                               "device": latency["device"]}
        comparison_specs = {
            2: [("gbm_router_vs_hybrid", "router_gbm", "hybrid"),
                ("gbm_router_vs_random", "router_gbm", "random_gbm")],
            3: [("skip_router_vs_hybrid", "router", "hybrid"),
                ("skip_router_vs_random", "router", "random")],
            4: [("predicted_gain_vs_random_10pct", "predicted_gain@0.1", "random@0.1"),
                ("confidence_vs_random_10pct", "router_confidence@0.1", "random@0.1"),
                ("always_rerank_vs_hybrid", "reranked", "hybrid")],
            "tau": [("router_tau_vs_hybrid_10pct", "router_tau@0.1", "hybrid"),
                    ("router_tau_vs_random_10pct", "router_tau@0.1", "router_random@0.1")],
        }
        for exp_id, specs in comparison_specs.items():
            frame = held[exp_id]
            subset = frame[frame["dataset"] == ds] if frame is not None and "dataset" in frame else None
            for label, score_a, score_b in specs:
                result = _paired_comparison(subset, score_a, score_b)
                if result:
                    comparisons[label] = result
        if strong_held is not None:
            subset = strong_held[strong_held["dataset"] == ds] if "dataset" in strong_held else None
            for label, a_col, b_col in (
                ("strong_predicted_gain_vs_random_10pct", "predicted_gain@0.1", "random@0.1"),
                ("strong_confidence_vs_random_10pct", "router_confidence@0.1", "random@0.1"),
                ("strong_always_rerank_vs_hybrid", "reranked", "hybrid"),
            ):
                result = _paired_comparison(subset, a_col, b_col)
                if result:
                    comparisons[label] = result
        unavailable = []
        if baseline is None:
            unavailable.extend(["BM25/dense/RRF baseline metrics", "MRR@10/MAP/Recall@100"])
        if held[3] is None or not (held[3]["dataset"] == ds).any():
            unavailable.append("skip_dense")
        if held[4] is None or not (held[4]["dataset"] == ds).any():
            unavailable.append("Experiment 4 escalation")
        if (strong_held is None or "dataset" not in strong_held or not (strong_held["dataset"] == ds).any()) and not benchmark_ds:
            unavailable.append("stronger-reranker Experiment 4")
        unavailable.append("advantage_regression_router")
        if not any(key[0] == ds for key in latency_by):
            unavailable.append("measured policy latency")
        ds_headroom = (headroom or {}).get("per_dataset", {}).get(ds)
        out = {"dataset": ds, "primary_metric": "ndcg_cut_10", "systems": systems,
               "headroom_vs_hybrid": ({"oracle_minus_hybrid": ds_headroom["oracle3_minus_hybrid"],
                                       "noise_corrected": ds_headroom["oracle3_minus_noise"],
                                       "is_upper_bound_only": True} if ds_headroom else None),
               "paired_comparisons": comparisons,
               "router_confusion_matrices": _router_confusion(root, ds, held[2], summaries[2])
                    if summaries[2] else None,
               "strong_reranker": ({"model": strong_summary.get("cross_encoder_model"),
                                    "device": strong_summary.get("device")}
                                   if strong_summary else
                                   {"models": list(benchmark_ds.get("models", {})),
                                    "device": (reranker_benchmark or {}).get("settings", {}).get("device")}
                                   if benchmark_ds else None),
               "secondary_metrics": "Available for baseline systems when baseline artifacts exist.",
               "unavailable_systems": unavailable,
               "source": "saved baseline summaries and held-out Experiment 2–4 artifacts"}
        folder = args.output_dir / ds
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        tables[ds] = systems
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Combined held-out evaluation", "", "Primary metric: mean nDCG@10.",
             "Baseline metrics come from BM25/dense/hybrid/RRF runs; adaptive systems come from held-out Experiment 2–4 artifacts. Matched-pool reranker results are included when a benchmark summary is supplied. Missing systems are listed as unavailable.",
             "95% CIs are bootstrap intervals over held-out queries where per-query values are saved. Latency is measured "
             "per query (results/policy_latency) where the dataset is available locally.",
             "", "| Dataset | System | nDCG@10 | 95% CI | Mean / p95 ms | Queries |",
             "| :--- | :--- | ---: | :---: | ---: | ---: |"]
    for ds, systems in tables.items():
        for name, val in systems.items():
            ci = val.get("ndcg_10", {}).get("ci_95") if isinstance(val.get("ndcg_10"), dict) else None
            ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—"
            lat = val.get("latency_ms")
            lat_text = f"{lat['mean']:.1f} / {lat['p95']:.1f}" if lat and "p95" in lat else "—"
            lines.append(f"| {ds} | {name} | {val['ndcg_10_mean']:.4f} | {ci_text} | {lat_text} | {val['n']} |")
    lines += ["", "## Paired held-out comparisons", "",
              "Bootstrap confidence intervals and paired Wilcoxon tests use per-query nDCG@10 values.",
              "Baseline artifacts also report MRR@10, MAP, and Recall@100.",
              "", "| Dataset | Comparison | Δ nDCG@10 [95% bootstrap CI] | Bootstrap p | Wilcoxon p | n |",
              "| :--- | :--- | :--- | ---: | ---: | ---: |"]
    for ds, systems in tables.items():
        comparisons = json.loads((args.output_dir / ds / "metrics.json").read_text())["paired_comparisons"]
        for label, result in comparisons.items():
            b = result["paired_bootstrap"]
            w = result.get("paired_wilcoxon")
            wilcoxon_text = f"{w['p_value']:.4f}" if w else "n/a"
            lines.append(f"| {ds} | {label} | {b['delta_mean']:+.4f} [{b['ci_95_low']:+.4f}, {b['ci_95_high']:+.4f}] "
                         f"| {b['p_value']:.4f} | {wilcoxon_text} | {result['n']} |")
    lines += ["", "## Pairwise win/loss/tie counts", "",
              "Counts compare per-query nDCG@10; each label is system A versus system B.",
              "", "| Dataset | Comparison | Wins | Losses | Ties | n |",
              "| :--- | :--- | ---: | ---: | ---: | ---: |"]
    for ds in tables:
        comparisons = json.loads((args.output_dir / ds / "metrics.json").read_text())["paired_comparisons"]
        for label, result in comparisons.items():
            if not result.get("win_loss_tie"):
                continue
            counts = result["win_loss_tie"]
            lines.append(f"| {ds} | {label} | {counts['wins']} | {counts['losses']} | {counts['ties']} | {counts['n']} |")
    lines += ["", "## Router confusion matrices", "",
              "Rows are observed best-route labels using each held-out fold's α and ε; columns are router predictions.",
              "Label order: BM25, dense, hybrid.", ""]
    for ds in tables:
        metrics = json.loads((args.output_dir / ds / "metrics.json").read_text())
        matrices = metrics["router_confusion_matrices"]
        if not matrices:
            continue
        for router, matrix in matrices.items():
            lines.append(f"- **{ds}, {router}:** `{matrix['confusion_matrix']}`")
    unavailable = sorted({name for ds in tables for name in
                          json.loads((args.output_dir / ds / "metrics.json").read_text())["unavailable_systems"]})
    lines += ["", "Unavailable from current artifacts: " + ", ".join(unavailable) + "."]
    (args.output_dir / "combined.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
