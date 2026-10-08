"""Run BM25-only, dense-only, hybrid-sweep, and RRF baselines."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval baselines (BM25, dense, hybrid sweep, RRF)")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., nfcorpus, scifact, fiqa)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/metrics"),
        help="Directory to save baseline results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Running retrieval baselines for dataset: {args.dataset}")
    # TODO: Run BM25, Dense, Hybrid alpha sweep, and RRF evaluations
    raise NotImplementedError


if __name__ == "__main__":
    main()
