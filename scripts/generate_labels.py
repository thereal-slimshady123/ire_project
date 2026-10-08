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
    # TODO: Evaluate routes, apply margin epsilon, compute regret, save labels parquet
    raise NotImplementedError


if __name__ == "__main__":
    main()
