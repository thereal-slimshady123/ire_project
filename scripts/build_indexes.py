"""Build BM25 and FAISS indexes for a dataset."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS indexes for a dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., nfcorpus, scifact, fiqa)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/indexes"),
        help="Directory to save generated index files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/retrievers.yaml"),
        help="Path to retrievers config YAML",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Building indexes for dataset: {args.dataset}")
    # TODO: Build BM25 and FAISS dense indexes and serialize to output_dir
    raise NotImplementedError


if __name__ == "__main__":
    main()
