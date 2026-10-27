"""Compute risk-coverage points from held-out confidence and regret arrays.

Input CSV columns are `confidence` (larger is more reliable) and `regret` for the
cheap decision. Supply held-out predictions; thresholds must not be tuned on this file.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from router_retrieval.evaluation.risk_coverage import compute_risk_coverage_curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("results/metrics"))
    p.add_argument("--taus", type=float, nargs="+", default=[.35, .4, .45, .5, .55, .6, .65, .7])
    args = p.parse_args()
    data = pd.read_csv(args.input)
    if not {"confidence", "regret"}.issubset(data.columns):
        raise ValueError("input must contain confidence and regret columns")
    curve = compute_risk_coverage_curve(data.confidence.to_numpy(), data.regret.to_numpy(), args.taus)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.output_dir / "risk_coverage.csv", index=False)
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
