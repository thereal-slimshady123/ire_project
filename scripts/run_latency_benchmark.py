"""Dedicated latency benchmark measuring mean and p95 per component."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run dedicated latency benchmark for pipeline components")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., nfcorpus, scifact, fiqa)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/latency"),
        help="Directory to save latency benchmark data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Running latency benchmark on dataset: {args.dataset}")
    # TODO: Benchmark per query (mean, p95): BM25, dense encode+search, hybrid fusion, feature
    #       extraction, router inference, cross-encoder rerank (50 candidates). Report totals for
    #       BM25-only, hybrid, router (cheap path), router (escalated path) and always-rerank.
    raise NotImplementedError


if __name__ == "__main__":
    main()
