"""Exploratory matched-pool comparison of configurable cross-encoder checkpoints.

This benchmark compares each reranker with the same fold-selected hybrid candidate
pools. It does not replace the pre-registered MiniLM-L6 result in Experiment 4.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from router_retrieval.data.splits import create_kfold_splits
from router_retrieval.evaluation.ranking_metrics import compute_ranking_metrics, ndcg_at_k
from router_retrieval.evaluation.significance import paired_bootstrap_test
from router_retrieval.retrieval.bm25 import document_text
from router_retrieval.retrieval.cross_encoder import CrossEncoderReranker
from router_retrieval.utils.config import load_config


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp1 = _module("exp1", "run_experiment1_hybrid_degradation.py")
exp1b = _module("exp1b", "run_experiment1b_query_types.py")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["nfcorpus", "scifact", "fiqa"])
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output_dir", type=Path, default=Path("results/reranker_benchmark"))
    p.add_argument("--dense_model", default="intfloat/e5-small-v2")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_bootstrap", type=int, default=10000)
    args = p.parse_args()
    encfg = load_config(ROOT / "configs" / "cross_encoder.yaml")
    ecfg = load_config(ROOT / "configs" / "experiment.yaml")
    retcfg = load_config(ROOT / "configs" / "retrievers.yaml")
    hcfg = load_config(ROOT / "configs" / "hybrid.yaml")
    models = args.models or encfg["candidate_models"]
    alphas = [float(x) for x in hcfg["alpha_grid"]]
    rows, summaries = [], {}
    for ds in args.datasets:
        data_path = args.processed_dir / "quora-sub" if ds == "quora" else args.data_dir / ds
        corpus, queries, qrels = exp1b.load_split(data_path)
        run_args = argparse.Namespace(results_dir=args.results_dir, dense_model=args.dense_model)
        br, dr = exp1b.get_runs(ds, corpus, queries, run_args, retcfg,
                                retcfg["dense"].get("top_k", 100))
        for q in queries:
            br.setdefault(q, [])
            dr.setdefault(q, [])
        k = retcfg["dense"].get("top_k", 100)
        hybs = exp1.build_hybrid_runs(br, dr, alphas, k, hcfg["normalization"]["method"])
        metrics = {a: compute_ranking_metrics(qrels, hybs[a], ["ndcg_cut_10"])["ndcg_cut_10"]
                   for a in alphas}
        fold_alphas = [max(alphas, key=lambda a: metrics[a].loc[list(tr)].mean())
                       for tr, _ in create_kfold_splits(list(queries), ecfg["k_folds"], args.seed)]
        fold_by_q = {}
        for i, (_, te) in enumerate(create_kfold_splits(list(queries), ecfg["k_folds"], args.seed)):
            for q in te:
                fold_by_q[q] = i
        base = {q: ndcg_at_k([c.doc_id for c in hybs[fold_alphas[fold_by_q[q]]][q]], qrels[q], 10)
                for q in queries}
        summaries[ds] = {"hybrid": float(sum(base.values()) / len(base)), "models": {}}
        for model_name in models:
            reranker = CrossEncoderReranker(model_name=model_name, batch_size=encfg["batch_size"],
                                            device=args.device, candidate_pool_size=50,
                                            max_length=encfg["max_length"])
            values = {}
            latencies = []
            first_query = next(iter(queries))
            reranker.score_pairs([(queries[first_query].text,
                                   document_text(corpus[hybs[fold_alphas[fold_by_q[first_query]]][first_query][0].doc_id]))])
            for q in queries:
                alpha = fold_alphas[fold_by_q[q]]
                candidates = hybs[alpha][q]
                start = time.perf_counter()
                final = reranker.rerank(queries[q], candidates, corpus, top_k=100)
                latencies.append((time.perf_counter() - start) * 1000.0)
                values[q] = ndcg_at_k(final, qrels[q], 10)
            a = [values[q] for q in sorted(values)]
            b = [base[q] for q in sorted(values)]
            comp = paired_bootstrap_test(a, b, args.n_bootstrap, args.seed)
            entry = {"n": len(values), "mean_ndcg10": float(sum(a) / len(a)),
                     "hybrid_ndcg10": summaries[ds]["hybrid"], "reranker_minus_hybrid": comp,
                     "rerank_latency_ms_mean": float(sum(latencies) / len(latencies)),
                     "rerank_latency_ms_p50": float(np.percentile(latencies, 50)),
                     "rerank_latency_ms_p95": float(np.percentile(latencies, 95))}
            summaries[ds]["models"][model_name] = entry
            rows.append({"dataset": ds, "model": model_name, **entry})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"exploratory": True, "results": summaries}, indent=2))
    lines = ["# Cross-encoder checkpoint benchmark", "",
             "Exploratory matched-candidate-pool comparison. The registered Experiment 4 remains the primary result.",
             "", "| Dataset | Model | Queries | Hybrid nDCG@10 | Reranker nDCG@10 | Δ [95% paired bootstrap CI] | Mean / p95 ms |",
             "| :--- | :--- | ---: | ---: | ---: | :--- | :--- |"]
    for row in rows:
        d = row["reranker_minus_hybrid"]
        lines.append(f"| {row['dataset']} | {row['model']} | {row['n']} | {row['hybrid_ndcg10']:.4f} "
                     f"| {row['mean_ndcg10']:.4f} | {d['delta_mean']:+.4f} "
                     f"[{d['ci_95_low']:+.4f}, {d['ci_95_high']:+.4f}] | "
                     f"{row['rerank_latency_ms_mean']:.1f} / {row['rerank_latency_ms_p95']:.1f} |")
    (args.output_dir / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
