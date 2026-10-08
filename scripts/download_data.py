"""Fetch BEIR datasets into data/raw/."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download BEIR datasets to data/raw/")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nfcorpus", "scifact", "fiqa"],
        help="List of BEIR dataset names to download",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save raw datasets",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Downloading datasets: {args.datasets} to {args.output_dir}")
    # TODO: Fetch and unpack datasets from BEIR repository URLs
    raise NotImplementedError


if __name__ == "__main__":
    main()
