"""Cost-quality frontier of skipping dense retrieval: measured latency x held-out nDCG@10 at matched skip rates.

Per query, every component is timed once after warm-up on the chosen device:
  bm25_search, dense_encode (real e5 query encoder), faiss_search, hybrid_fusion,
  skip_feature_extraction + router_inference (Exp 3 LightGBM, same features/settings), h1_score (BM25 top1/top2 ratio).
FAISS IndexFlatIP search is timed over random unit vectors with the dataset's corpus size and the encoder's dimension
(flat search cost depends only on size x dimension), so corpora need not be re-encoded.

Policy latency at skip rate r (expected value over which queries are skipped):
  always_hybrid = bm25 + encode + faiss + fusion
  random        = bm25 + (1 - r) * (encode + faiss + fusion)
  H1            = bm25 + h1_score + (1 - r) * (encode + faiss + fusion)
  router        = bm25 + skip_features + router_inference + (1 - r) * (encode + faiss + fusion)
Quality at r comes from the matched-rate per-query file of scripts/run_e4_matched_rate.py.
Break-even skip rate = per-query decision overhead / (encode + faiss + fusion).

Usage:
  python scripts/run_cost_frontier.py --datasets <ds ...> --exp2_dir <dir ...> --matched_dir <dir> --device cpu --output_dir <dir>
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
    compute_query_term_coverage, compute_score_entropy, compute_top_gap, compute_top_ratio,
)
from router_retrieval.data.schema import Candidate  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
from router_retrieval.retrieval.bm25 import BM25Retriever, document_text, tokenize  # noqa: E402
from router_retrieval.retrieval.dense import DenseRetriever  # noqa: E402
from router_retrieval.retrieval.hybrid import BM25_TRUNCATED_FLOOR, fuse_normalized_scores, normalize_for_fusion  # noqa: E402
from router_retrieval.retrieval.normalization import min_max_normalize  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp1b = _load("exp1b", "run_experiment1b_query_types.py")
exp3 = _load("exp3", "run_experiment3_skip_dense.py")

COMPONENTS = ["bm25_search", "dense_encode", "faiss_search", "hybrid_fusion", "skip_feature_extraction",
              "router_inference", "h1_score"]
RATES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]


def time_dataset(ds, args, ret_cfg, router_cfg, seed):
    import faiss

    corpus, queries, _ = exp1b.load_split(ROOT / "data" / "raw" / ds)
    src = next(d for d in args.exp2_dir if (d / "cache" / f"{ds}.csv").exists())
    table = exp3.load_table(ds, argparse.Namespace(exp2_dir=src, results_dir=ROOT / "results"))
    k = ret_cfg["dense"].get("top_k", 100)
    bm25 = BM25Retriever(k1=ret_cfg["bm25"]["k1"], b=ret_cfg["bm25"]["b"])
    bm25.index(corpus)
    idf_vocab = {t: float(bm25.idf[j]) for t, j in bm25.vocab.items()}
    vocab, rare = set(bm25.vocab), rare_idf_threshold(bm25.num_docs)
    prefixes = exp1b.E5_PREFIXES if "e5" in args.dense_model.lower() else {}
    dense = DenseRetriever(model_name=args.dense_model, batch_size=1, device=args.device,
                           faiss_index_type="IndexFlatIP", max_length=ret_cfg["dense"].get("max_length"), **prefixes)
    dim = dense.encode([dense.query_prefix + "warm up"]).shape[1]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((len(corpus), dim)).astype("float32")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    doc_ids = sorted(corpus)

    # Timing-only router: Exp 3's classifier and label, fit on this dataset (only its inference time is used).
    alpha = max([float(c.split("_")[-1]) for c in table if c.startswith("ndcg_hyb_")],
                key=lambda a: table[f"ndcg_hyb_{a}"].mean())
    y = (table["ndcg_bm25"] >= table[f"ndcg_hyb_{alpha}"] - exp3.LABEL_MARGIN).astype(int).values
    router = exp3.fit_classifier(table[exp3.SKIP_FEATURES], y, router_cfg, seed)

    def run_query(q):
        t, clock = {}, time.perf_counter
        s = clock(); b = bm25.search(q, k); t["bm25_search"] = clock() - s
        s = clock(); emb = dense.encode([dense.query_prefix + q.text]); t["dense_encode"] = clock() - s
        s = clock()
        sc, ix = index.search(np.ascontiguousarray(emb, dtype="float32"), k)
        d = [Candidate(q.query_id, doc_ids[i], float(v), r + 1) for r, (v, i) in enumerate(zip(sc[0], ix[0]))]
        t["faiss_search"] = clock() - s
        s = clock()
        fuse_normalized_scores(q.query_id, normalize_for_fusion(b, k, "min_max", BM25_TRUNCATED_FLOOR),
                               normalize_for_fusion(d, k, "min_max"), alpha, k)
        t["hybrid_fusion"] = clock() - s
        s = clock()
        f = extract_query_features(q, idf_vocab, vocab, rare_threshold=rare)
        f.update({"bm25_top1_score": b[0].score if b else 0.0, "bm25_top1_top2_gap": compute_top_gap(min_max_normalize(b)),
                  "bm25_top1_top2_ratio": compute_top_ratio(b), "bm25_score_entropy": compute_score_entropy(b),
                  "bm25_top10_query_term_coverage": compute_query_term_coverage(
                      tokenize(q.text), [set(tokenize(document_text(corpus[c.doc_id]))) for c in b[:10]]),
                  "bm25_num_hits": len(b)})
        feats = pd.DataFrame([f])[exp3.SKIP_FEATURES]
        t["skip_feature_extraction"] = clock() - s
        s = clock(); exp3.predict_p(router, feats); t["router_inference"] = clock() - s
        s = clock(); compute_top_ratio(b); t["h1_score"] = clock() - s
        return {n: 1000.0 * v for n, v in t.items()}

    qlist = [queries[q] for q in table.index if q in queries]
    for q in qlist[: args.warmup]:
        run_query(q)
    rows = {q.query_id: run_query(q) for q in qlist}
    out = pd.DataFrame.from_dict(rows, orient="index")[COMPONENTS]
    out.insert(0, "dataset", ds)
    return out, len(corpus), dim


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--exp2_dir", type=Path, nargs="+", required=True)
    ap.add_argument("--matched_dir", type=Path, required=True, help="output dir of run_e4_matched_rate.py")
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--dense_model", default="intfloat/e5-small-v2")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()
    cfg = lambda n: load_config(ROOT / "configs" / n)  # noqa: E731
    ret_cfg, router_cfg, seed = cfg("retrievers.yaml"), cfg("router.yaml"), cfg("experiment.yaml")["seed"]

    quality = pd.read_csv(args.matched_dir / "per_query_heldout.csv", index_col=0)
    timings, meta = [], {}
    for ds in args.datasets:
        t, n_docs, dim = time_dataset(ds, args, ret_cfg, router_cfg, seed)
        timings.append(t)
        meta[ds] = {"n_docs": n_docs, "dim": dim}
        print(f"[{ds}] {args.device}: " + ", ".join(f"{c}={t[c].mean():.2f}" for c in COMPONENTS))
    timings = pd.concat(timings)

    def frontier(t: pd.DataFrame, qd: pd.DataFrame) -> dict:
        m = t[COMPONENTS].mean()
        dense_part = m["dense_encode"] + m["faiss_search"] + m["hybrid_fusion"]
        over = {"router": m["skip_feature_extraction"] + m["router_inference"], "H1": m["h1_score"], "random": 0.0}
        rows = []
        for r in RATES:
            key = f"{r:.2f}"
            q = lambda m_: float(qd["hybrid"].mean()) if r == 0 else float(qd[f"{m_}@{key}"].mean())  # noqa: E731
            rows.append({"skip_rate": r, "hybrid_ms": m["bm25_search"] + dense_part,
                         **{f"{p}_ms": m["bm25_search"] + over[p] + (1 - r) * dense_part for p in over},
                         "hybrid_ndcg": float(qd["hybrid"].mean()), "router_ndcg": q("router"),
                         "H1_ndcg": q("H1_bm25_top_ratio"), "random_ndcg": q("random")})
        b = lambda a, c: paired_bootstrap_test(qd[a].values, qd[c].values, 10000, seed)  # noqa: E731
        tests = {"H1_minus_hybrid@0.20": b("H1_bm25_top_ratio@0.20", "hybrid"),
                 "H1_minus_random@0.20": b("H1_bm25_top_ratio@0.20", "random@0.20"),
                 "router_minus_hybrid@0.20": b("router@0.20", "hybrid")}
        return {"components_ms": m.to_dict(), "dense_part_ms": dense_part, "tests": tests,
                "latency_saving_H1@0.20": 1 - (m["bm25_search"] + over["H1"] + 0.8 * dense_part) / (m["bm25_search"] + dense_part),
                "break_even_skip_rate": {p: (over[p] / dense_part) for p in over}, "frontier": rows}

    res = {ds: frontier(timings[timings.dataset == ds], quality[quality.dataset == ds]) for ds in args.datasets}
    res["pooled"] = frontier(timings, quality[quality.dataset.isin(args.datasets)])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    timings.to_csv(out / f"timings_{args.device}.csv")
    (out / f"summary_{args.device}.json").write_text(json.dumps({"device": args.device, "meta": meta, **res}, indent=2),
                                                     encoding="utf-8")
    L = [f"# Cost–quality frontier of skipping dense retrieval ({args.device})", "",
         f"Datasets: {', '.join(args.datasets)}. Dense model {args.dense_model}; FAISS flat search timed over random unit "
         "vectors of the corpus size. Quality: held-out nDCG@10 at matched skip rates (run_e4_matched_rate.py).", "",
         "## Mean component latency (ms)", "", "| Dataset | " + " | ".join(COMPONENTS) + " | Break-even skip rate: router / H1 |",
         "| :--- | " + " | ".join([":---:"] * (len(COMPONENTS) + 1)) + " |"]
    for ds, v in res.items():
        L.append(f"| {ds} | " + " | ".join(f"{v['components_ms'][c]:.2f}" for c in COMPONENTS)
                 + f" | {v['break_even_skip_rate']['router']:.1%} / {v['break_even_skip_rate']['H1']:.2%} |")
    L += ["", "## Pooled frontier", "", "| Skip rate | Hybrid ms | Router ms | H1 ms | Random ms | Router nDCG | H1 nDCG | Random nDCG | Hybrid nDCG |",
          "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for r in res["pooled"]["frontier"]:
        L.append(f"| {r['skip_rate']:.0%} | {r['hybrid_ms']:.2f} | {r['router_ms']:.2f} | {r['H1_ms']:.2f} | {r['random_ms']:.2f} "
                 f"| {r['router_ndcg']:.4f} | {r['H1_ndcg']:.4f} | {r['random_ndcg']:.4f} | {r['hybrid_ndcg']:.4f} |")
    ci = lambda t: f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"  # noqa: E731
    L += ["", "## Quality at 20% skip rate (paired bootstrap)", "",
          "| Dataset | H1 − hybrid | H1 − random | Router − hybrid | H1 latency saving vs hybrid |", "| :--- | :---: | :---: | :---: | :---: |"]
    for ds, v in res.items():
        t = v["tests"]
        L.append(f"| {ds} | {ci(t['H1_minus_hybrid@0.20'])} | {ci(t['H1_minus_random@0.20'])} | {ci(t['router_minus_hybrid@0.20'])} "
                 f"| {v['latency_saving_H1@0.20']:.1%} |")
    (out / f"tables_{args.device}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
