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
    # TODO: Benchmark BM25, Dense encode+search, feature extraction, router inference, cross-encoder
    raise NotImplementedError


if __name__ == "__main__":
    main()
