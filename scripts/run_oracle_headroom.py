"""Report the recorded per-query oracle headroom from Experiment 1."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--results_dir", type=Path, default=Path("results/experiment1"))
    p.add_argument("--threshold", type=float, default=0.02)
    args = p.parse_args()
    path = args.results_dir / args.dataset / "summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    h = data["H2"]
    ci = h["oracle3_vs_hybrid"]
    report = {"dataset": args.dataset, "n": data["n_queries"], "hybrid_ndcg10": h["hybrid_ndcg"],
              "oracle_ndcg10": h["oracle3_ndcg"], "headroom": h["headroom"],
              "ci_95": [ci["ci_95_low"], ci["ci_95_high"]], "bootstrap_p": ci["p_value"],
              "is_upper_bound_only": True, "meets_descriptive_threshold": h["headroom"] >= args.threshold}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
