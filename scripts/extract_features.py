"""Extract query and retrieval features into parquet."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features for queries and save to parquet")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., nfcorpus, scifact, fiqa)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/features"),
        help="Directory to save features parquet",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Extracting features for dataset: {args.dataset}")
    # TODO: Extract query features + retrieval signals, persist features dataframe
    raise NotImplementedError


if __name__ == "__main__":
    main()
