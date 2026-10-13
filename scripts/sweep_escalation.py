"""Sweep confidence threshold tau and compute risk-coverage curves."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep confidence threshold tau to evaluate risk-coverage trade-offs")
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
        help="Directory to save sweep results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Sweeping escalation thresholds for dataset: {args.dataset}")
    # TODO: Using the trained router: compute the risk-coverage curve and routed nDCG@10 vs.
    #       escalation rate on test, with tau taken from the DEV calibration for each rate.
    raise NotImplementedError


if __name__ == "__main__":
    main()
