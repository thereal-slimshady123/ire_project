"""Extract query and retrieval-signal features into Parquet (plan.md Contract 2).

Two sources:
  runs   Recompute from the raw BEIR split and the cached BM25 / dense runs (`results/runs/{dataset}/`,
         TREC or Contract-1 Parquet) with `FeatureBuilder`. Needs `data/raw` and the runs.
  cache  Re-export the per-query features already stored in `results/experiment2/cache/{dataset}.csv`
         (works on a fresh clone, no raw data needed).
  auto   `runs` when the raw data and both runs are present, otherwise `cache`.

Usage:
  python scripts/extract_features.py --dataset scifact fiqa
  python scripts/extract_features.py --source runs --verify_against_cache
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.runs import read_run_parquet, read_trec_run  # noqa: E402
from router_retrieval.features.feature_builder import FEATURE_COLUMNS, FeatureBuilder  # noqa: E402
from router_retrieval.retrieval.bm25 import BM25Retriever  # noqa: E402
from router_retrieval.router.cv import load_route_table  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402

DATASETS = ["nfcorpus", "scifact", "fiqa", "cqadupstack-unix", "quora"]
log = get_logger("extract_features")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features for queries and save to parquet")
    parser.add_argument("--dataset", "--datasets", dest="datasets", nargs="+", default=DATASETS,
                        help="Dataset name(s) (default: all five evaluation datasets)")
    parser.add_argument("--source", choices=["auto", "runs", "cache"], default="auto")
    parser.add_argument("--output_dir", type=Path, default=Path("results/features"),
                        help="Directory to save features parquet")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--config_dir", type=Path, default=Path("configs"))
    parser.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    parser.add_argument("--quora_queries", type=int, default=1000)
    parser.add_argument("--quora_distractors", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify_against_cache", action="store_true",
                        help="With --source runs: report max |difference| to the Experiment 2 cache per feature")
    return parser.parse_args()


def _find_run(run_dir: Path, stem_glob: str) -> Optional[Path]:
    for pattern in (f"{stem_glob}.parquet", f"{stem_glob}.trec"):
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def _read_run(path: Path):
    return read_run_parquet(path) if path.suffix == ".parquet" else read_trec_run(path)


def available_run_paths(args, dataset: str, ret_cfg) -> Optional[Tuple[Path, Path]]:
    run_dir = args.results_dir / "runs" / dataset
    bm25 = _find_run(run_dir, "bm25*")
    dense = _find_run(run_dir, f"dense_{args.dense_model.replace('/', '__')}") or _find_run(run_dir, "dense*")
    return (bm25, dense) if bm25 and dense else None


def features_from_runs(args, dataset: str, ret_cfg) -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("exp1b", ROOT / "scripts" / "run_experiment1b_query_types.py")
    exp1b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp1b)

    if dataset == "quora":
        path = exp1b.build_quora_subset(args.data_dir / "quora", args.processed_dir / "quora-sub",
                                        args.quora_queries, args.quora_distractors, args.seed)
    else:
        path = args.data_dir / dataset
    corpus, queries, _ = exp1b.load_split(path)
    bm25_path, dense_path = available_run_paths(args, dataset, ret_cfg)
    bm25_run, dense_run = _read_run(bm25_path), _read_run(dense_path)
    bm25 = BM25Retriever(k1=ret_cfg["bm25"]["k1"], b=ret_cfg["bm25"]["b"])
    bm25.index(corpus)  # IDF statistics only
    builder = FeatureBuilder.from_bm25(bm25, corpus)
    ordered = [queries[q] for q in sorted(queries)]
    return builder.build_feature_matrix(ordered, bm25_run, dense_run)


def verify(features: pd.DataFrame, dataset: str, results_dir: Path) -> None:
    cached = load_route_table(dataset, results_dir, use_features_parquet=False)
    common = cached.index.intersection(features.index)
    log.info(f"[{dataset}] verify vs cache on {len(common)}/{len(cached)} queries")
    for col in FEATURE_COLUMNS:
        diff = float(np.abs(features.loc[common, col].to_numpy() - cached.loc[common, col].to_numpy()).max())
        log.info(f"    {col:<34s} max|diff| = {diff:.3g}" + ("   <-- CHECK" if diff > 1e-4 else ""))


def main():
    args = parse_args()
    ret_cfg = load_config(args.config_dir / "retrievers.yaml")
    for ds in args.datasets:
        use_runs = args.source == "runs" or (
            args.source == "auto" and (args.data_dir / ds).exists() and available_run_paths(args, ds, ret_cfg))
        if use_runs:
            log.info(f"[{ds}] extracting features from raw data + cached runs")
            feats = features_from_runs(args, ds, ret_cfg)
            if args.verify_against_cache:
                verify(feats, ds, args.results_dir)
        else:
            log.info(f"[{ds}] re-exporting features from the Experiment 2 cache")
            feats = load_route_table(ds, args.results_dir, use_features_parquet=False)[FEATURE_COLUMNS]
        out = args.output_dir / f"{ds}.parquet"
        FeatureBuilder().save_features(feats, out)
        log.info(f"[{ds}] wrote {out} ({len(feats)} queries x {len(FEATURE_COLUMNS)} features)")


if __name__ == "__main__":
    main()
