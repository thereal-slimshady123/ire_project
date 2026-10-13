"""Compute oracle router headroom over best fixed-alpha hybrid (Week-2 checkpoint).

Computed on train + dev queries only (never test). Because the oracle takes a per-query max,
part of its headroom is rank noise; the noise floor is the same oracle computed between
hybrid(alpha*) and hybrid(alpha* +/- 0.1), which have no real systematic difference.
Gate: (headroom - noise_floor) >= threshold. No significance test vs hybrid is run, since
oracle >= hybrid on every query by construction.
"""

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
        help="Gate threshold on (headroom - noise floor) in nDCG@10 (default 0.02 / 2 points)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Computing oracle router headroom for dataset: {args.dataset}")
    # TODO: On train + dev queries: headroom = oracle{bm25, dense, hybrid(alpha*)} - hybrid(alpha*);
    #       noise_floor = oracle{hybrid(alpha*), hybrid(alpha*-0.1), hybrid(alpha*+0.1)} - hybrid(alpha*).
    #       If headroom - noise_floor < threshold, flag the dataset for re-scoping.
    raise NotImplementedError


if __name__ == "__main__":
    main()
