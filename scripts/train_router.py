"""Train logistic and GBM query routers."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train logistic regression and gradient boosted decision tree routers")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., nfcorpus, scifact, fiqa)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/models"),
        help="Directory to save trained model artifacts",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Training routers for dataset: {args.dataset}")
    # TODO: Train LogisticRouter and GBMRouter using k-fold CV and regret weighting
    raise NotImplementedError


if __name__ == "__main__":
    main()
