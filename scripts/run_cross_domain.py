"""Cross-domain transfer evaluation: train router on dataset X, test on dataset Y."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run cross-domain router generalization study")
    parser.add_argument(
        "--source_dataset",
        type=str,
        required=True,
        help="Dataset name to train router on",
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        required=True,
        help="Dataset name to test router on",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/metrics"),
        help="Directory to save cross-domain results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Running cross-domain generalization: {args.source_dataset} -> {args.target_dataset}")
    # TODO: Use the router (and tau) trained on the source dataset's train/dev; evaluate routing
    #       on the target dataset's TEST queries (zero-shot: the source's alpha* is used for hybrid).
    raise NotImplementedError


if __name__ == "__main__":
    main()
