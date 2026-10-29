"""Cross-domain transfer evaluation: train router on dataset X, test zero-shot on dataset Y.

For every ordered pair (X -> Y) the advantage router is fitted on all of X's queries (alpha*, theta and
temperature from X only) and applied to all of Y's queries. We report routed nDCG@10 minus
  (a) always-hybrid with X's alpha*  -- the zero-shot comparison a deployment would face, and
  (b) always-hybrid with Y's in-domain CV alpha,
next to the in-domain (5-fold CV) result on Y, so the "transfer penalty" is the difference of the two
router-minus-hybrid numbers. A leave-one-dataset-out row (train on the other four) is added per target.

Usage:
  python scripts/run_cross_domain.py
  python scripts/run_cross_domain.py --source_dataset fiqa --target_dataset scifact
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
from router_retrieval.router.cv import (  # noqa: E402
    FEATURE_GROUPS, cv_predict, fit_predict_fold, fmt_ci, load_route_table, prefix_index,
)
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402

DATASETS = ["nfcorpus", "scifact", "fiqa", "cqadupstack-unix", "quora"]
log = get_logger("cross_domain")


def parse_args():
    parser = argparse.ArgumentParser(description="Run cross-domain router generalization study")
    parser.add_argument("--source_dataset", type=str, default=None,
                        help="Dataset name to train router on (default: every dataset)")
    parser.add_argument("--target_dataset", type=str, default=None,
                        help="Dataset name to test router on (default: every dataset)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Universe of datasets for the all-pairs sweep and the leave-one-dataset-out pool")
    parser.add_argument("--features", default="all", choices=sorted(FEATURE_GROUPS))
    parser.add_argument("--kind", default="advantage", choices=["advantage", "advantage_fixed"],
                        help="advantage: calibrated theta + guard; advantage_fixed: theta fixed at 0.02")
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiment2/cross_domain"),
                        help="Directory to save cross-domain results")
    parser.add_argument("--use_features_parquet", action="store_true",
                        help="Read features from results/features/{dataset}.parquet (Contract 2) instead of the cache")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--config_dir", type=Path, default=Path("configs"))
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    return parser.parse_args()


def transfer_row(source_name, target_name, out, in_domain_gap, in_domain_hybrid, n_boot, seed) -> dict:
    vs_source = paired_bootstrap_test(out["routed"].to_numpy(), out["hybrid"].to_numpy(), n_boot, seed)
    vs_indomain_alpha = paired_bootstrap_test(out["routed"].to_numpy(), in_domain_hybrid, n_boot, seed)
    return {
        "source": source_name, "target": target_name, "n": int(len(out)),
        "hybrid_source_alpha": float(out["hybrid"].mean()),
        "hybrid_indomain_alpha": float(in_domain_hybrid.mean()),
        "routed": float(out["routed"].mean()),
        "routed_minus_hybrid_source_alpha": vs_source,
        "routed_minus_hybrid_indomain_alpha": vs_indomain_alpha,
        "indomain_routed_minus_hybrid": in_domain_gap,
        "transfer_penalty": in_domain_gap - vs_source["delta_mean"],
        "theta_bm25": out.attrs.get("theta_bm25"), "theta_dense": out.attrs.get("theta_dense"),
        "route_share": {r: float((out["route_predicted"] == r).mean()) for r in ("bm25", "dense", "hybrid")},
    }


def write_tables(rows, path: Path, kind: str = "advantage") -> None:
    lines = ["# Cross-domain transfer of the advantage router (`" + kind + "`)", "",
             "Router fitted on the source dataset(s) only (alpha*, theta, temperature, regressors), applied "
             "zero-shot to every query of the target. 'vs hybrid(src alpha)' compares with always-hybrid at the "
             "source's alpha*; 'in-domain' is the 5-fold CV result on the target; 'penalty' = in-domain gap minus "
             "transfer gap. Differences are nDCG@10 [95% bootstrap CI].", "",
             "| Source | Target | n | Hybrid(src α) | Routed | Routed − hybrid(src α) | Routed − hybrid(in-domain α) | In-domain routed − hybrid | Penalty | Shares B/D/H |",
             "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for r in rows:
        sh = "/".join(f"{r['route_share'][k]:.0%}" for k in ("bm25", "dense", "hybrid"))
        lines.append(
            f"| {r['source']} | {r['target']} | {r['n']} | {r['hybrid_source_alpha']:.4f} | {r['routed']:.4f} "
            f"| {fmt_ci(r['routed_minus_hybrid_source_alpha'])} | {fmt_ci(r['routed_minus_hybrid_indomain_alpha'])} "
            f"| {r['indomain_routed_minus_hybrid']:+.4f} | {r['transfer_penalty']:+.4f} | {sh} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")
    router_cfg = load_config(args.config_dir / "router.yaml")
    seed, k_folds = exp_cfg["seed"], exp_cfg["k_folds"]
    seed_everything(seed)
    alphas = [float(a) for a in hyb_cfg["alpha_grid"]]
    cols = FEATURE_GROUPS[args.features]
    sources = [args.source_dataset] if args.source_dataset else args.datasets
    targets = [args.target_dataset] if args.target_dataset else args.datasets
    needed = [*sources, *targets, *([] if args.source_dataset else args.datasets)]
    tables = {ds: load_route_table(ds, args.results_dir, args.use_features_parquet) for ds in dict.fromkeys(needed)}

    # In-domain 5-fold CV for each target: the reference the transfer penalty is measured against.
    indomain = {}
    for ds in targets:
        held, _ = cv_predict(tables[ds], args.kind, cols, alphas, router_cfg, k_folds, seed)
        gain = float((held["routed"] - held["hybrid"]).mean())
        indomain[ds] = (held, gain)

    rows = []
    for tgt in targets:
        held, gap = indomain[tgt]
        for src in sources:
            if src == tgt:
                continue
            out, info = fit_predict_fold(tables[src], tables[tgt], args.kind, cols, alphas, router_cfg, seed)
            out.attrs.update(theta_bm25=info["theta_bm25"], theta_dense=info["theta_dense"])
            rows.append(transfer_row(src, tgt, out, gap, held["hybrid"].to_numpy(), args.n_bootstrap, seed))
            r = rows[-1]
            log.info(f"[{src} -> {tgt}] routed-hybrid(src α) = {fmt_ci(r['routed_minus_hybrid_source_alpha'])}  "
                     f"(in-domain {gap:+.4f})")
        if not args.source_dataset:  # leave-one-dataset-out
            pool = pd.concat([prefix_index(tables[n], n) for n in args.datasets if n != tgt])
            out, info = fit_predict_fold(pool, tables[tgt], args.kind, cols, alphas, router_cfg, seed)
            out.attrs.update(theta_bm25=info["theta_bm25"], theta_dense=info["theta_dense"])
            rows.append(transfer_row("(other four)", tgt, out, gap, held["hybrid"].to_numpy(), args.n_bootstrap, seed))
            log.info(f"[others -> {tgt}] routed-hybrid(src α) = {fmt_ci(rows[-1]['routed_minus_hybrid_source_alpha'])}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat = [{**{k: v for k, v in r.items() if not isinstance(v, dict)},
             "gap_src_alpha": r["routed_minus_hybrid_source_alpha"]["delta_mean"],
             "gap_src_alpha_ci_low": r["routed_minus_hybrid_source_alpha"]["ci_95_low"],
             "gap_src_alpha_ci_high": r["routed_minus_hybrid_source_alpha"]["ci_95_high"]} for r in rows]
    pd.DataFrame(flat).to_csv(args.output_dir / "matrix.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    write_tables(rows, args.output_dir / "tables.md", args.kind)
    log.info(f"Wrote {args.output_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
