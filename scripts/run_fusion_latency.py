"""Measured per-query latency of E6's per-document learned fusion, on top of the hybrid pipeline (descriptive).

Learned fusion runs BM25 and dense exactly like the hybrid, then replaces min-max fusion with:
  query_features   the 18 router features (same code path as run_policy_latency.py's feature block)
  candidate_feats  per-document features over the BM25 + dense top-100 union (run_learned_fusion.candidate_table logic)
  rank             LightGBM LambdaRank scoring of the candidates + sort
Every component is timed once per query after warm-up, on CPU. The model is retrained exactly as in E6 (same data,
same deterministic settings); only its inference time is used. BM25 / dense runs come from the cached run files, so
retrieval itself is not re-timed here: add the hybrid latency from run_cost_frontier.py's timings.

Usage:
  python scripts/run_fusion_latency.py --datasets <ds ...> --exp2_dir <dir> --train_datasets <ds ...> \
      --train_exp2_dir <dir ...> --cost_timings <timings_cpu.csv> --output_dir <dir>
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from router_retrieval.features.query_features import extract_query_features, rare_idf_threshold  # noqa: E402
from router_retrieval.features.retrieval_signal_features import (  # noqa: E402
    compute_query_term_coverage, compute_top_ratio, extract_retrieval_signal_features,
)
from router_retrieval.retrieval.bm25 import BM25Retriever, document_text, tokenize  # noqa: E402
from router_retrieval.retrieval.hybrid import BM25_TRUNCATED_FLOOR, normalize_for_fusion  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp1 = _load("exp1", "run_experiment1_hybrid_degradation.py")
exp1b = _load("exp1b", "run_experiment1b_query_types.py")
lf = _load("lf", "run_learned_fusion.py")
K = 100


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--exp2_dir", type=Path, required=True)
    ap.add_argument("--train_datasets", nargs="+", required=True)
    ap.add_argument("--train_exp2_dir", type=Path, nargs="+", required=True)
    ap.add_argument("--cost_timings", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()
    lf.FEATS = lf.FEATURE_SETS["doc+query"]
    router_cfg = load_config(ROOT / "configs" / "router.yaml")
    ret_cfg = load_config(ROOT / "configs" / "retrievers.yaml")

    train = []
    for ds in args.train_datasets:
        src = next(d for d in args.train_exp2_dir if (d / "cache" / f"{ds}.csv").exists())
        c = pd.read_csv(src / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        c.index = c.index.astype(str)
        train.append(lf.candidate_table(ds, c, "intfloat__e5-small-v2"))
    model = lf.fit(pd.concat(train), router_cfg)

    cost = pd.read_csv(args.cost_timings, index_col=0)
    rows = []
    for ds in args.datasets:
        corpus, queries, _ = exp1b.load_split(ROOT / "data" / "raw" / ds)
        bm25 = BM25Retriever(k1=ret_cfg["bm25"]["k1"], b=ret_cfg["bm25"]["b"])
        bm25.index(corpus)
        idf_vocab = {t: float(bm25.idf[j]) for t, j in bm25.vocab.items()}
        vocab, rare = set(bm25.vocab), rare_idf_threshold(bm25.num_docs)
        run_dir = ROOT / "results" / "runs" / ds
        brun = exp1.read_trec_run(run_dir / "bm25_k1-1.2_b-0.75.trec")
        drun = exp1.read_trec_run(run_dir / "dense_intfloat__e5-small-v2.trec")
        cache = pd.read_csv(args.exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        qids = [q for q in cache.index.astype(str) if q in queries]

        def one(q):
            t, clock = {}, time.perf_counter
            b, d = brun.get(q, []), drun.get(q, [])
            s = clock()
            f = extract_query_features(queries[q], idf_vocab, vocab, rare_threshold=rare)
            f.update(extract_retrieval_signal_features(b, d))
            f["bm25_top1_top2_ratio"] = compute_top_ratio(b)
            qt = tokenize(queries[q].text)
            f["bm25_top10_query_term_coverage"] = compute_query_term_coverage(qt, [set(tokenize(document_text(corpus[c.doc_id]))) for c in b[:10]])
            f["dense_top10_query_term_coverage"] = compute_query_term_coverage(qt, [set(tokenize(document_text(corpus[c.doc_id]))) for c in d[:10]])
            t["query_features"] = clock() - s
            s = clock()
            bn, dn = normalize_for_fusion(b, K, "min_max", BM25_TRUNCATED_FLOOR), normalize_for_fusion(d, K, "min_max")
            br, dr = {c.doc_id: c.rank for c in b}, {c.doc_id: c.rank for c in d}
            docs = sorted(set(br) | set(dr))
            X = pd.DataFrame({"b_norm": [bn.get(x, 0.0) for x in docs], "d_norm": [dn.get(x, 0.0) for x in docs],
                              "b_rank": [br.get(x, K + 1) for x in docs], "d_rank": [dr.get(x, K + 1) for x in docs],
                              "in_b": [int(x in br) for x in docs], "in_d": [int(x in dr) for x in docs]})
            X["rank_diff"] = X["b_rank"] - X["d_rank"]
            for k in lf.exp2.FEATURES:
                X[k] = f.get(k, 0.0)
            t["candidate_features"] = clock() - s
            s = clock()
            order = np.argsort(-model.predict(X[lf.FEATS]), kind="stable")
            _ = [docs[i] for i in order[:K]]
            t["rank"] = clock() - s
            return {k: 1000.0 * v for k, v in t.items()} | {"n_candidates": len(docs)}

        for q in qids[: args.warmup]:
            one(q)
        r = pd.DataFrame({q: one(q) for q in qids}).T
        r.insert(0, "dataset", ds)
        rows.append(r)
        print(f"[{ds}] " + ", ".join(f"{c}={r[c].median():.2f}" for c in ["query_features", "candidate_features", "rank"]))
    t = pd.concat(rows)
    t["fusion_overhead_ms"] = t[["query_features", "candidate_features", "rank"]].sum(axis=1)
    cost["hybrid_ms"] = cost[["bm25_search", "dense_encode", "faiss_search", "hybrid_fusion"]].sum(axis=1)
    summ = {}
    for ds, g in [*t.groupby("dataset", sort=False), ("pooled", t)]:
        cg = cost if ds == "pooled" else cost[cost.dataset == ds]
        h_med, o_med = float(cg["hybrid_ms"].median()), float(g["fusion_overhead_ms"].median())
        summ[ds] = {"hybrid_median_ms": h_med, "fusion_overhead_median_ms": o_med,
                    "learned_fusion_median_ms": h_med - float(cg["hybrid_fusion"].median()) + o_med,
                    "overhead_p95_ms": float(g["fusion_overhead_ms"].quantile(0.95)),
                    "components_median_ms": {c: float(g[c].median()) for c in ["query_features", "candidate_features", "rank"]},
                    "median_candidates": float(g["n_candidates"].median())}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "fusion_timings_cpu.csv")
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    L = ["# Learned-fusion latency (CPU, descriptive)", "", "Medians (robust to the one suspended measurement in the hybrid timings).", "",
         "| Dataset | Hybrid ms | + query features | + candidate features | + LambdaRank | Learned fusion ms | Overhead p95 | Candidates |",
         "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for ds, v in summ.items():
        c = v["components_median_ms"]
        L.append(f"| {ds} | {v['hybrid_median_ms']:.2f} | {c['query_features']:.2f} | {c['candidate_features']:.2f} | {c['rank']:.2f} "
                 f"| {v['learned_fusion_median_ms']:.2f} | {v['overhead_p95_ms']:.2f} | {v['median_candidates']:.0f} |")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
