"""Generate per-query route labels and regret weights into parquet."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate per-route nDCG labels with margin threshold")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., nfcorpus, scifact, fiqa)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/labels"),
        help="Directory to save labels parquet",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Generating labels for dataset: {args.dataset}")
    # TODO: For train, dev and test queries: per-route nDCG@10 (hybrid at alpha* from train),
    #       margin labels for every epsilon in configs/labels.yaml, and per-route regret.
    #       Save labels/{dataset}.parquet: query_id, split, epsilon, best_route, margin,
    #       top1_score, top2_score, ndcg_bm25, ndcg_dense, ndcg_hybrid,
    #       regret_bm25, regret_dense, regret_hybrid. Test labels are for analysis only.
    raise NotImplementedError


if __name__ == "__main__":
    main()
