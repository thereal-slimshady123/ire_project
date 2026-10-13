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
    # TODO: On TEST queries only: BM25, dense, hybrid(alpha*), RRF, always-rerank, router at each
    #       target escalation rate, random escalation (router routes, same count), always-hybrid +
    #       random rerank (same count), oracle router, and oracle escalation (ceiling).
    #       Metrics: nDCG@10, MRR@10, MAP@100, Recall@100. Paired bootstrap of the router vs each
    #       non-oracle system with Holm-Bonferroni correction per dataset.
    raise NotImplementedError


if __name__ == "__main__":
    main()
