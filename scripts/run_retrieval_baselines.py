"""Run retrieval baselines: BM25, Dense, Hybrid (5-fold CV alpha sweep), and RRF."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.beir_loader import load_beir_dataset
from router_retrieval.data.schema import Candidate, Query, Document
from router_retrieval.data.splits import create_kfold_splits
from router_retrieval.evaluation.ranking_metrics import compute_ranking_metrics, aggregate_metrics
from router_retrieval.evaluation.significance import paired_bootstrap_test
from router_retrieval.retrieval.bm25 import BM25Retriever
from router_retrieval.retrieval.dense import DenseRetriever
from router_retrieval.retrieval.hybrid import (
    BM25_TRUNCATED_FLOOR,
    fuse_normalized_scores,
    normalize_for_fusion,
)
from router_retrieval.retrieval.rrf import ReciprocalRankFusion
from router_retrieval.utils.config import load_config
from router_retrieval.utils.logging_utils import get_logger
from router_retrieval.utils.seeding import seed_everything

log = get_logger("retrieval_baselines")
DEFAULT_DATASETS = ["nfcorpus", "scifact", "fiqa"]
METRICS = ["ndcg_cut_10", "mrr_10", "map", "recall_100"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run retrieval baselines (BM25, Dense, Tuned Hybrid CV sweep, RRF)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="List of datasets to evaluate (e.g. nfcorpus scifact fiqa cqadupstack-unix quora)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Root data directory",
    )
    parser.add_argument(
        "--config_dir",
        type=Path,
        default=Path("configs"),
        help="Directory with retriever and hybrid YAML configs",
    )
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=Path("results/runs"),
        help="Directory to read/write candidate pool runs",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/metrics"),
        help="Directory to save baseline metrics and alpha sweeps",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Candidate pool size",
    )
    parser.add_argument(
        "--dense_model",
        type=str,
        default=None,
        help="Dense embedding model override (default from configs/retrievers.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('auto', 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for significance testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def read_trec_run(path: Path) -> Dict[str, List[Candidate]]:
    """Read a standard TREC run file."""
    run: Dict[str, List[Candidate]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, did, rank, score, _ = parts[:6]
                run.setdefault(str(qid), []).append(
                    Candidate(query_id=str(qid), doc_id=str(did), score=float(score), rank=int(rank))
                )
    return run


def write_trec_run(run: Dict[str, List[Candidate]], path: Path, tag: str) -> None:
    """Write run to TREC run format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, cands in run.items():
            for c in cands:
                f.write(f"{qid} Q0 {c.doc_id} {c.rank} {c.score:.6f} {tag}\n")


def save_run_parquet(run: Dict[str, List[Candidate]], path: Path) -> None:
    """Save candidate run conforming to Contract 1 schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for qid, cands in run.items():
        for c in cands:
            rows.append(
                {
                    "query_id": str(c.query_id),
                    "doc_id": str(c.doc_id),
                    "score": np.float32(c.score),
                    "rank": np.int32(c.rank),
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def get_dataset_dir(data_dir: Path, dataset: str) -> Path:
    """Locate dataset directory."""
    processed = data_dir / "processed" / f"{dataset}-sub"
    if processed.exists():
        return processed
    raw = data_dir / "raw" / dataset
    if raw.exists():
        return raw
    return data_dir / dataset


def get_or_create_runs(
    dataset: str,
    corpus: Dict[str, Document],
    queries: Dict[str, Query],
    args,
    ret_cfg: dict,
) -> Tuple[Dict[str, List[Candidate]], Dict[str, List[Candidate]]]:
    """Retrieve or load cached runs for BM25 and Dense."""
    run_dir = args.runs_dir / dataset
    run_dir.mkdir(parents=True, exist_ok=True)

    bm25_cfg = ret_cfg.get("bm25", {"k1": 1.2, "b": 0.75})
    k1, b = float(bm25_cfg.get("k1", 1.2)), float(bm25_cfg.get("b", 0.75))
    bm25_trec = run_dir / f"bm25_k1-{k1}_b-{b}.trec"
    bm25_parquet = run_dir / "bm25.parquet"

    dense_cfg = ret_cfg.get("dense", {"model_name": "intfloat/e5-small-v2"})
    model_name = args.dense_model or dense_cfg.get("model_name", "intfloat/e5-small-v2")
    device = args.device or dense_cfg.get("device", "auto")
    slug = model_name.replace("/", "__")
    dense_trec = run_dir / f"dense_{slug}.trec"
    dense_parquet = run_dir / "dense.parquet"

    # BM25 Run
    if bm25_trec.exists():
        log.info(f"[{dataset}] Loading existing BM25 run from {bm25_trec}...")
        bm25_run = read_trec_run(bm25_trec)
    else:
        log.info(f"[{dataset}] Computing BM25 run for {len(queries)} queries...")
        bm25 = BM25Retriever(k1=k1, b=b)
        bm25.index(corpus)
        bm25_run = bm25.batch_search([queries[q] for q in sorted(queries)], k=args.k)
        write_trec_run(bm25_run, bm25_trec, "bm25")

    save_run_parquet(bm25_run, bm25_parquet)

    # Dense Run
    cached_dense = list(run_dir.glob("dense_*.trec"))
    if dense_trec.exists():
        log.info(f"[{dataset}] Loading existing Dense run from {dense_trec}...")
        dense_run = read_trec_run(dense_trec)
    elif cached_dense:
        dense_trec = cached_dense[0]
        log.info(f"[{dataset}] Loading existing Dense run from {dense_trec}...")
        dense_run = read_trec_run(dense_trec)
    else:
        index_dir = Path("results/indexes") / dataset / slug
        passage_prefix = "passage: " if "e5" in model_name.lower() else ""
        query_prefix = "query: " if "e5" in model_name.lower() else ""
        kwargs = dict(
            model_name=model_name,
            batch_size=dense_cfg.get("batch_size", 64),
            device=device,
            passage_prefix=passage_prefix,
            query_prefix=query_prefix,
        )
        if (index_dir / "embeddings.npy").exists():
            log.info(f"[{dataset}] Loading FAISS dense index from {index_dir}...")
            dense = DenseRetriever.load(index_dir, **kwargs)
        else:
            log.info(f"[{dataset}] Indexing FAISS dense embeddings with {model_name}...")
            dense = DenseRetriever(**kwargs)
            dense.index(corpus, show_progress_bar=True)
            dense.save(index_dir)

        log.info(f"[{dataset}] Searching dense index for {len(queries)} queries...")
        dense_run = dense.batch_search([queries[q] for q in sorted(queries)], k=args.k)
        write_trec_run(dense_run, dense_trec, slug)

    save_run_parquet(dense_run, dense_parquet)

    # Ensure all test queries are present
    for qid in queries:
        bm25_run.setdefault(qid, [])
        dense_run.setdefault(qid, [])

    return bm25_run, dense_run


def evaluate_dataset(dataset: str, args, ret_cfg: dict, hyb_cfg: dict) -> dict:
    log.info(f"========== Benchmarking: {dataset} ==========")
    ds_dir = get_dataset_dir(args.data_dir, dataset)
    corpus, queries, qrels = load_beir_dataset(ds_dir, split="test")
    qlist = sorted(queries.keys())
    log.info(f"[{dataset}] Loaded {len(corpus)} docs, {len(queries)} test queries, {len(qrels)} qrels.")

    bm25_run, dense_run = get_or_create_runs(dataset, corpus, queries, args, ret_cfg)

    # 1. Evaluate BM25
    df_bm25 = compute_ranking_metrics(qrels, bm25_run, metrics=METRICS)
    bm25_agg = aggregate_metrics(df_bm25)

    # 2. Evaluate Dense
    df_dense = compute_ranking_metrics(qrels, dense_run, metrics=METRICS)
    dense_agg = aggregate_metrics(df_dense)

    # 3. Sweep Hybrid alphas
    alphas = [float(a) for a in hyb_cfg.get("alpha_grid", np.linspace(0.0, 1.0, 11))]
    norm_method = hyb_cfg.get("normalization", {}).get("method", "min_max")
    k = args.k

    norm_b = {q: normalize_for_fusion(bm25_run[q], k, norm_method, BM25_TRUNCATED_FLOOR) for q in qlist}
    norm_d = {q: normalize_for_fusion(dense_run[q], k, norm_method) for q in qlist}

    hybrid_runs: Dict[float, Dict[str, List[Candidate]]] = {}
    alpha_dfs: Dict[float, pd.DataFrame] = {}
    sweep_rows = []

    for a in alphas:
        run_a = {q: fuse_normalized_scores(q, norm_b[q], norm_d.get(q, {}), a, k) for q in qlist}
        hybrid_runs[a] = run_a
        df_a = compute_ranking_metrics(qrels, run_a, metrics=METRICS)
        alpha_dfs[a] = df_a
        agg = aggregate_metrics(df_a)
        sweep_rows.append({"alpha": a, **agg})

    sweep_df = pd.DataFrame(sweep_rows)
    best_in_sample_idx = sweep_df["ndcg_cut_10"].idxmax()
    best_in_sample_alpha = float(sweep_df.loc[best_in_sample_idx, "alpha"])

    # 5-fold CV to pick non-strawman out-of-fold alpha*
    seed = args.seed
    n_folds = 5
    oof_hybrid_run: Dict[str, List[Candidate]] = {}
    oof_ndcgs = {}
    fold_alphas = []

    for train_q, test_q in create_kfold_splits(qlist, n_folds, seed):
        fold_train_ndcg = {a: alpha_dfs[a].loc[train_q, "ndcg_cut_10"].mean() for a in alphas}
        best_alpha = max(fold_train_ndcg, key=fold_train_ndcg.get)
        fold_alphas.append(float(best_alpha))
        for q in test_q:
            oof_hybrid_run[q] = hybrid_runs[best_alpha][q]
            oof_ndcgs[q] = alpha_dfs[best_alpha].loc[q, "ndcg_cut_10"]

    df_hybrid_cv = compute_ranking_metrics(qrels, oof_hybrid_run, metrics=METRICS)
    hybrid_cv_agg = aggregate_metrics(df_hybrid_cv)

    # Save Hybrid Contract 1 run
    save_run_parquet(oof_hybrid_run, args.runs_dir / dataset / "hybrid.parquet")

    # 4. Evaluate RRF
    rrf_k = int(hyb_cfg.get("rrf", {}).get("k", 60))
    rrf = ReciprocalRankFusion(k=rrf_k)
    rrf_run = {q: rrf.fuse([bm25_run[q], dense_run[q]], top_k=k) for q in qlist}
    df_rrf = compute_ranking_metrics(qrels, rrf_run, metrics=METRICS)
    rrf_agg = aggregate_metrics(df_rrf)

    # Save RRF Contract 1 run
    save_run_parquet(rrf_run, args.runs_dir / dataset / "rrf.parquet")

    # Significance testing vs Tuned Hybrid CV
    sig_bm25 = paired_bootstrap_test(
        df_bm25["ndcg_cut_10"].values, df_hybrid_cv["ndcg_cut_10"].values, args.n_bootstrap, seed
    )
    sig_dense = paired_bootstrap_test(
        df_dense["ndcg_cut_10"].values, df_hybrid_cv["ndcg_cut_10"].values, args.n_bootstrap, seed
    )
    sig_rrf = paired_bootstrap_test(
        df_rrf["ndcg_cut_10"].values, df_hybrid_cv["ndcg_cut_10"].values, args.n_bootstrap, seed
    )

    # Format summary JSON conforming to Contract 5
    summary = {
        "dataset": dataset,
        "n_queries": len(queries),
        "bm25": {
            **bm25_agg,
            "vs_hybrid": sig_bm25,
        },
        "dense": {
            **dense_agg,
            "vs_hybrid": sig_dense,
        },
        "hybrid_cv": {
            **hybrid_cv_agg,
            "fold_alphas": fold_alphas,
            "mean_alpha": float(np.mean(fold_alphas)),
        },
        "hybrid_insample_best": {
            "alpha": best_in_sample_alpha,
            "ndcg_cut_10": float(sweep_df.loc[best_in_sample_idx, "ndcg_cut_10"]),
        },
        "rrf": {
            "k": rrf_k,
            **rrf_agg,
            "vs_hybrid": sig_rrf,
        },
    }

    # Save outputs
    out_dir = args.output_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "baselines.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sweep_df.to_csv(out_dir / "alpha_sweep.csv", index=False)

    log.info(
        f"[{dataset}] Baseline nDCG@10: BM25={bm25_agg['ndcg_cut_10']:.4f}, "
        f"Dense={dense_agg['ndcg_cut_10']:.4f}, Hybrid(CV)={hybrid_cv_agg['ndcg_cut_10']:.4f}, "
        f"RRF={rrf_agg['ndcg_cut_10']:.4f}"
    )

    return summary


def main():
    args = parse_args()
    seed_everything(args.seed)
    ret_cfg = load_config(args.config_dir / "retrievers.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")

    all_summaries = {}
    for ds in args.datasets:
        all_summaries[ds] = evaluate_dataset(ds, args, ret_cfg, hyb_cfg)

    # Print master markdown table
    lines = [
        "",
        "# Retrieval Baselines Summary (5-fold CV Hybrid & RRF)",
        "",
        "| Dataset | Queries | BM25 nDCG@10 | Dense nDCG@10 | Hybrid (CV) nDCG@10 | RRF nDCG@10 | Best Hybrid alpha |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for ds, s in all_summaries.items():
        lines.append(
            f"| `{ds}` | {s['n_queries']} | {s['bm25']['ndcg_cut_10']:.4f} | "
            f"{s['dense']['ndcg_cut_10']:.4f} | **{s['hybrid_cv']['ndcg_cut_10']:.4f}** | "
            f"{s['rrf']['ndcg_cut_10']:.4f} | {s['hybrid_insample_best']['alpha']:.1f} |"
        )
    table_str = "\n".join(lines)
    print(table_str)

    (args.output_dir / "baselines_summary.md").write_text(table_str + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

