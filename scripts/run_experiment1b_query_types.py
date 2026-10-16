"""Experiment 1b: for specific query types, does always-hybrid lose to BM25 or dense?

Mixed workload: CQADupStack-unix (technical questions: error strings, flags, paths) and a
subsample of Quora (conversational, paraphrastic questions). One global alpha is tuned by k-fold
CV over the pooled queries -- what "always use hybrid" means for a single deployed system -- and
each query searches its own corpus. Query types are fixed by a-priori rules (TECH_RE below) before
any per-type result is computed; within each type we run paired tests of fixed systems
(BM25 vs Hybrid, Dense vs Hybrid), which avoids the per-query max-of-two selection bias.

Usage:
  python scripts/run_experiment1b_query_types.py
"""

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.beir_loader import load_corpus, load_qrels, load_queries, qrels_to_dict  # noqa: E402
from router_retrieval.evaluation.ranking_metrics import compute_ranking_metrics  # noqa: E402
from router_retrieval.retrieval.bm25 import BM25Retriever  # noqa: E402
from router_retrieval.retrieval.dense import DenseRetriever  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp1", ROOT / "scripts" / "run_experiment1_hybrid_degradation.py")
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)

log = get_logger("experiment1b")
PRIMARY = "ndcg_cut_10"
METRICS = ["ndcg_cut_10", "mrr_10", "recall_100"]

# A-priori "technical identifier" rule, frozen before looking at any per-type result: a flag,
# path, file name with a known extension, glob, shell variable, underscore identifier, quoted
# string, or a token mixing letters and digits.
TECH_PATTERNS = [
    r"(?<!\w)--?[A-Za-z][\w-]*",
    r"(?:^|\s)~?/[\w.\-/]+",
    r"\b\w+\.(?:sh|py|conf|txt|log|d|so|ko|cfg|ini|json|xml|deb|rpm|gz|tar|zip|iso|img)\b",
    r"\*\.\w+",
    r"\$\{?\w+",
    r"\b\w*_\w+\b",
    r"[\"'`][^\"'`]{2,}[\"'`]",
    r"\b(?=\w*\d)(?=\w*[A-Za-z])\w+\b",
]
TECH_RE = re.compile("|".join(TECH_PATTERNS))

E5_PREFIXES = {"query_prefix": "query: ", "passage_prefix": "passage: "}


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 1b: per-query-type hybrid vs single retrievers")
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output_dir", type=Path, default=Path("results/experiment1b"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--quora_queries", type=int, default=1000, help="Quora test queries to sample")
    p.add_argument("--quora_distractors", type=int, default=50000, help="Random non-relevant Quora docs kept")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--epsilon", type=float, default=0.02)
    return p.parse_args()


def classify_type(dataset: str, text: str) -> str:
    if dataset == "quora":
        return "quora_conversational"
    return "unix_technical" if TECH_RE.search(text) else "unix_plain"


def build_quora_subset(src: Path, dst: Path, n_queries: int, n_distractors: int, seed: int) -> Path:
    """Sample test queries; keep their relevant docs plus random distractors (deterministic)."""
    if (dst / "corpus.jsonl").exists():
        return dst
    rng = random.Random(seed)
    qrels = qrels_to_dict(load_qrels(src / "qrels" / "test.tsv"))
    qids = rng.sample(sorted(qrels), n_queries)
    relevant = {d for q in qids for d in qrels[q]}
    with open(src / "corpus.jsonl", encoding="utf-8") as f:
        lines = f.readlines()
    rel_lines, other = [], []
    for line in lines:
        (rel_lines if json.loads(line)["_id"] in relevant else other).append(line)
    keep = rel_lines + rng.sample(other, min(n_distractors, len(other)))
    queries_out = [l for l in open(src / "queries.jsonl", encoding="utf-8") if json.loads(l)["_id"] in set(qids)]

    (dst / "qrels").mkdir(parents=True, exist_ok=True)
    (dst / "corpus.jsonl").write_text("".join(keep), encoding="utf-8")
    (dst / "queries.jsonl").write_text("".join(queries_out), encoding="utf-8")
    with open(dst / "qrels" / "test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q in qids:
            for d, r in qrels[q].items():
                f.write(f"{q}\t{d}\t{r}\n")
    log.info(f"[quora] subset: {len(qids)} queries, {len(rel_lines)} relevant + {len(keep) - len(rel_lines)} distractor docs")
    return dst


def load_split(path: Path):
    corpus = load_corpus(path / "corpus.jsonl")
    qrels = qrels_to_dict(load_qrels(path / "qrels" / "test.tsv"))
    queries = {q: v for q, v in load_queries(path / "queries.jsonl").items() if q in qrels}
    return corpus, queries, {q: qrels[q] for q in queries}


def get_runs(name: str, corpus, queries, args, ret_cfg, k: int):
    run_dir = args.results_dir / "runs" / name
    qlist = [queries[q] for q in sorted(queries)]
    bm25_cfg = ret_cfg["bm25"]
    bm25_path = run_dir / f"bm25_k1-{bm25_cfg['k1']}_b-{bm25_cfg['b']}.trec"
    if bm25_path.exists():
        bm25_run = exp1.read_trec_run(bm25_path)
    else:
        log.info(f"[{name}] BM25 over {len(corpus)} docs")
        bm25 = BM25Retriever(k1=bm25_cfg["k1"], b=bm25_cfg["b"])
        bm25.index(corpus)
        bm25_run = bm25.batch_search(qlist, k)
        exp1.write_trec_run(bm25_run, bm25_path, "bm25")

    slug = args.dense_model.replace("/", "__")
    dense_path = run_dir / f"dense_{slug}.trec"
    if dense_path.exists():
        dense_run = exp1.read_trec_run(dense_path)
    else:
        d = ret_cfg["dense"]
        prefixes = E5_PREFIXES if "e5" in args.dense_model.lower() else {}
        dense = DenseRetriever(model_name=args.dense_model, batch_size=d["batch_size"], device=d["device"],
                               faiss_index_type=d["faiss_index_type"], max_length=d.get("max_length"), **prefixes)
        log.info(f"[{name}] encoding {len(corpus)} docs with {args.dense_model} on {dense.device}")
        dense.index(corpus)
        dense_run = dense.batch_search(qlist, k)
        exp1.write_trec_run(dense_run, dense_path, slug)
    for q in queries:
        bm25_run.setdefault(q, [])
        dense_run.setdefault(q, [])
    return bm25_run, dense_run


def main():
    args = parse_args()
    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    ret_cfg = load_config(args.config_dir / "retrievers.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")
    seed, k = exp_cfg["seed"], ret_cfg["dense"].get("top_k", 100)
    seed_everything(seed)
    alphas = [float(a) for a in hyb_cfg["alpha_grid"]]

    splits = {
        "cqadupstack-unix": args.data_dir / "cqadupstack-unix",
        "quora": build_quora_subset(args.data_dir / "quora", args.processed_dir / "quora-sub",
                                    args.quora_queries, args.quora_distractors, seed),
    }

    rows = []
    for name, path in splits.items():
        corpus, queries, qrels = load_split(path)
        log.info(f"[{name}] {len(corpus)} docs, {len(queries)} queries")
        bm25_run, dense_run = get_runs(name, corpus, queries, args, ret_cfg, k)
        hyb = exp1.build_hybrid_runs(bm25_run, dense_run, alphas, k, hyb_cfg["normalization"]["method"])
        m_b = compute_ranking_metrics(qrels, bm25_run, METRICS)
        m_d = compute_ranking_metrics(qrels, dense_run, METRICS)
        m_a = {a: compute_ranking_metrics(qrels, hyb[a], METRICS) for a in alphas}
        for q in queries:
            row = {"query_id": f"{name}:{q}", "dataset": name, "text": queries[q].text,
                   "type": classify_type(name, queries[q].text)}
            for m in METRICS:
                row[f"{m}_bm25"] = m_b.at[q, m]
                row[f"{m}_dense"] = m_d.at[q, m]
                for a in alphas:
                    row[f"{m}_hyb_{a}"] = m_a[a].at[q, m]
            rows.append(row)

    pq = pd.DataFrame(rows).set_index("query_id")

    # "Always hybrid": ONE global alpha, chosen by CV on the pooled mixed workload (out-of-fold).
    alpha_ndcg = pq[[f"{PRIMARY}_hyb_{a}" for a in alphas]].copy()
    alpha_ndcg.columns = alphas
    oof_alpha, folds = exp1.select_alpha_cv(alpha_ndcg, exp_cfg["k_folds"], seed)
    for m in METRICS:
        pq[f"{m}_hybrid"] = [pq.at[q, f"{m}_hyb_{a}"] for q, a in oof_alpha.items()]
    pq["alpha_global_oof"] = oof_alpha

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {"dense_model": args.dense_model, "epsilon": args.epsilon, "cv_folds": folds,
               "alpha_star_global_insample": float(alpha_ndcg.mean().idxmax()), "types": {}}
    table = []
    for t, g in pq.groupby("type"):
        h, b, d = g[f"{PRIMARY}_hybrid"].values, g[f"{PRIMARY}_bm25"].values, g[f"{PRIMARY}_dense"].values
        per_type_alpha_means = {a: float(g[f"{PRIMARY}_hyb_{a}"].mean()) for a in alphas}
        best_type_alpha = max(per_type_alpha_means, key=per_type_alpha_means.get)
        entry = {
            "n": len(g),
            "mean": {"bm25": float(b.mean()), "dense": float(d.mean()), "hybrid_global": float(h.mean()),
                     "hybrid_best_alpha_for_type": per_type_alpha_means[best_type_alpha]},
            "best_alpha_for_type_insample": best_type_alpha,
            "alpha_curve": per_type_alpha_means,
            "bm25_vs_hybrid": exp1.paired_tests(b, h, args.n_bootstrap, seed),
            "dense_vs_hybrid": exp1.paired_tests(d, h, args.n_bootstrap, seed),
            "pairwise_share": {
                "bm25_beats_hybrid": float(np.mean(b - h >= args.epsilon)),
                "hybrid_beats_bm25": float(np.mean(h - b >= args.epsilon)),
                "dense_beats_hybrid": float(np.mean(d - h >= args.epsilon)),
                "hybrid_beats_dense": float(np.mean(h - d >= args.epsilon)),
            },
            "other_metrics_mean": {
                m: {r: float(g[f"{m}_{r}"].mean()) for r in ("bm25", "dense", "hybrid")} for m in METRICS
            },
        }
        summary["types"][t] = entry
        table.append((t, entry))

    pq.drop(columns=[c for c in pq.columns if "_hyb_" in c]).to_csv(out / "per_query.csv")
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    alphas_used = sorted({f["alpha_star"] for f in folds})
    lines = [
        "# Experiment 1b: query types where always-hybrid loses",
        "",
        f"Dense model: `{args.dense_model}`. Hybrid = one global α chosen by {len(folds)}-fold CV on the pooled "
        f"mixed workload (α* ∈ {{{', '.join(f'{a:g}' for a in alphas_used)}}}). nDCG@10; Δ = single − hybrid, "
        "paired bootstrap 95% CI and two-sided p; Wilcoxon one-sided (single > hybrid).",
        "",
        "| Query type | n | BM25 | Dense | Hybrid (global α) | BM25 − Hybrid [95% CI] (p) | Dense − Hybrid [95% CI] (p) "
        "| Best α for this type → nDCG |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for t, e in table:
        mb, md = e["bm25_vs_hybrid"], e["dense_vs_hybrid"]
        cell = lambda r: f"{r['delta_mean']:+.4f} [{r['ci_95_low']:+.4f}, {r['ci_95_high']:+.4f}] (p={r['p_value']:.1e}; W p={r['wilcoxon_p_greater']:.1e})"  # noqa: E731
        lines.append(
            f"| {t} | {e['n']} | {e['mean']['bm25']:.4f} | {e['mean']['dense']:.4f} | {e['mean']['hybrid_global']:.4f} "
            f"| {cell(mb)} | {cell(md)} | α={e['best_alpha_for_type_insample']:g} → {e['mean']['hybrid_best_alpha_for_type']:.4f} |"
        )
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for t, e in table:
        log.info(
            f"[{t}] n={e['n']} BM25={e['mean']['bm25']:.4f} Dense={e['mean']['dense']:.4f} "
            f"Hybrid(global)={e['mean']['hybrid_global']:.4f} | BM25-H={e['bm25_vs_hybrid']['delta_mean']:+.4f} "
            f"(p={e['bm25_vs_hybrid']['p_value']:.1e}) Dense-H={e['dense_vs_hybrid']['delta_mean']:+.4f} "
            f"(p={e['dense_vs_hybrid']['p_value']:.1e})"
        )
    log.info(f"Wrote {out / 'tables.md'}")


if __name__ == "__main__":
    main()
