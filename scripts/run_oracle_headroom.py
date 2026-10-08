"""Compute oracle router headroom over best fixed-alpha hybrid (Week-2 checkpoint)."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compute oracle vs hybrid headroom (Week-2 checkpoint gate)")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., nfcorpus, scifact, fiqa)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Headroom gate threshold in nDCG@10 (default 0.02 / 2 points)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Computing oracle router headroom for dataset: {args.dataset}")
    # TODO: Compare oracle routing vs. best fixed-alpha hybrid nDCG@10
    # Hard gate check: if headroom < threshold, alert to revisit dataset choice.
    raise NotImplementedError


if __name__ == "__main__":
    main()
