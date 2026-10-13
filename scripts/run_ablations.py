"""Feature ablation study: query-only vs. retrieval-signal vs. combined."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run router feature ablation experiments")
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
        help="Directory to save ablation evaluation results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Running feature ablations for dataset: {args.dataset}")
    # TODO: Train on train / select on dev / evaluate on test with query-only, retrieval-signal-only
    #       and combined feature sets (same protocol as train_router.py).
    raise NotImplementedError


if __name__ == "__main__":
    main()
