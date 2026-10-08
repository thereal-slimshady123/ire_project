"""Run full evaluation pipeline: router + all baselines and controls."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete evaluation across router, baselines, and controls")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nfcorpus", "scifact", "fiqa"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/metrics"),
        help="Directory to save final metrics tables",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Running full evaluation on datasets: {args.datasets}")
    # TODO: Evaluate BM25, Dense, Hybrid, RRF, Always-Rerank, Random Escalation, Router
    raise NotImplementedError


if __name__ == "__main__":
    main()
