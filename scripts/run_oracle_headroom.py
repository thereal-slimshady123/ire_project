"""Oracle-router headroom over tuned hybrid, with the per-query noise floor removed.

For every dataset in Experiment 2's per-query table (e5-small-v2 dense, all five datasets), each query is
scored with the alpha* of the training fold it was held out from. Reported, all as upper bounds:

  * oracle3   = max(BM25, dense, hybrid)            -- the routing oracle;
  * oracle2   = max(BM25, dense)                     -- an oracle that never fuses;
  * noise     = max(hybrid at alpha*-0.1, alpha*, alpha*+0.1) -- an "oracle" over three near-identical
                systems, i.e. how much a per-query max gains from rank jitter alone (issue A2);
  * headroom  = oracle3 - hybrid, and noise-corrected headroom = oracle3 - noise.

The proposal's week-2 gate (~2 nDCG points) is checked against the noise-corrected headroom.

Usage:
  python scripts/run_oracle_headroom.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402

DATASETS = ["nfcorpus", "scifact", "fiqa", "cqadupstack-unix", "quora"]


def headroom_frame(table: pd.DataFrame, fold_alpha: pd.Series, alpha_grid) -> pd.DataFrame:
    """Per-query hybrid, oracle and noise-oracle nDCG@10 at each query's held-out-fold alpha*."""
    grid = sorted(alpha_grid)
    rows = {}
    for q, a in fold_alpha.items():
        i = grid.index(a)
        neighbours = grid[max(i - 1, 0): i + 2]
        hyb = table.at[q, f"ndcg_hyb_{a}"]
        bm25, dense = table.at[q, "ndcg_bm25"], table.at[q, "ndcg_dense"]
        rows[q] = {"hybrid": hyb, "oracle3": max(bm25, dense, hyb), "oracle2": max(bm25, dense),
                   "noise": max(table.at[q, f"ndcg_hyb_{n}"] for n in neighbours)}
    return pd.DataFrame.from_dict(rows, orient="index")


def summarize(f: pd.DataFrame, n_boot: int, seed: int) -> dict:
    return {"n": len(f), "mean": {c: float(f[c].mean()) for c in f.columns},
            "oracle3_minus_hybrid": paired_bootstrap_test(f["oracle3"], f["hybrid"], n_boot, seed),
            "oracle2_minus_hybrid": paired_bootstrap_test(f["oracle2"], f["hybrid"], n_boot, seed),
            "noise_minus_hybrid": paired_bootstrap_test(f["noise"], f["hybrid"], n_boot, seed),
            "oracle3_minus_noise": paired_bootstrap_test(f["oracle3"], f["noise"], n_boot, seed),
            "is_upper_bound_only": True}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--exp2_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--output_dir", type=Path, default=Path("results/metrics"))
    p.add_argument("--threshold", type=float, default=0.02, help="Week-2 gate on noise-corrected headroom")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    args = p.parse_args()
    cfg = load_config(args.config_dir / "experiment.yaml")
    alphas = [float(a) for a in load_config(args.config_dir / "hybrid.yaml")["alpha_grid"]]
    summary = json.loads((args.exp2_dir / "summary.json").read_text(encoding="utf-8"))

    frames, report = [], {"per_dataset": {}}
    for ds in args.datasets:
        table = pd.read_csv(args.exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        table.index = table.index.astype(str)
        fold_alpha = pd.Series(index=table.index, dtype=float)
        for i, (_, test) in enumerate(create_kfold_splits(list(table.index), cfg["k_folds"], cfg["seed"])):
            fold_alpha.loc[test] = float(summary["fold_info"][ds][i]["alpha"])
        f = headroom_frame(table, fold_alpha, alphas)
        report["per_dataset"][ds] = summarize(f, args.n_bootstrap, cfg["seed"])
        frames.append(f)
    report["pooled"] = summarize(pd.concat(frames), args.n_bootstrap, cfg["seed"])
    for s in [*report["per_dataset"].values(), report["pooled"]]:
        s["meets_week2_gate_noise_corrected"] = s["oracle3_minus_noise"]["delta_mean"] >= args.threshold

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oracle_headroom.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ci = lambda t: f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"  # noqa: E731
    lines = ["# Oracle headroom over tuned hybrid (upper bounds)", "",
             "nDCG@10; dense = e5-small-v2; each query uses its held-out fold's training α*. The noise oracle picks "
             "the best of hybrid at α* − 0.1, α*, α* + 0.1, so it measures what a per-query max gains from rank "
             "jitter alone. Noise-corrected headroom = oracle − noise oracle. None of these is evidence that a "
             "router can realize the gain.", "",
             "| Dataset | n | Hybrid | Oracle | Oracle − hybrid | Oracle{BM25,dense} − hybrid | Noise oracle − hybrid "
             f"| Noise-corrected headroom | ≥ {args.threshold:.2f}? |",
             "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for name, s in [*report["per_dataset"].items(), ("**pooled**", report["pooled"])]:
        lines.append(f"| {name} | {s['n']} | {s['mean']['hybrid']:.4f} | {s['mean']['oracle3']:.4f} "
                     f"| {ci(s['oracle3_minus_hybrid'])} | {ci(s['oracle2_minus_hybrid'])} | {ci(s['noise_minus_hybrid'])} "
                     f"| {ci(s['oracle3_minus_noise'])} | {'yes' if s['meets_week2_gate_noise_corrected'] else 'no'} |")
    (args.output_dir / "oracle_headroom.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
