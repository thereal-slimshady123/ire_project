"""Feature ablation study: query-only vs. retrieval-signal vs. combined (and pre-dense) feature sets.

Each feature set is cross-validated with the Dual Advantage Regression router (and, for reference, the
margin-labelled GBM classifier) using exactly the Experiment 2 folds. We report held-out routed nDCG@10
minus always-hybrid with a paired bootstrap CI, per dataset and pooled, plus the gain importances of
regressors fitted on all of a dataset's queries.

Usage:
  python scripts/run_ablations.py
  python scripts/run_ablations.py --datasets scifact fiqa --kinds advantage
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.router.advantage_targets import advantage_targets, select_alpha  # noqa: E402
from router_retrieval.router.cv import (  # noqa: E402
    FEATURE_GROUPS, KINDS, advantage_kwargs, cv_predict, fmt_ci, load_route_table, prefix_index, summarize_heldout,
)
from router_retrieval.router.gbm_router import AdvantageRouter  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402

DATASETS = ["nfcorpus", "scifact", "fiqa", "cqadupstack-unix", "quora"]
GROUP_ORDER = ["query_only", "retrieval_signal", "pre_dense", "all"]
log = get_logger("ablations")


def parse_args():
    parser = argparse.ArgumentParser(description="Run router feature ablation experiments")
    parser.add_argument("--dataset", "--datasets", dest="datasets", nargs="+", default=DATASETS,
                        help="Dataset name(s) (default: all five evaluation datasets)")
    parser.add_argument("--groups", nargs="+", default=GROUP_ORDER, choices=sorted(FEATURE_GROUPS))
    parser.add_argument("--kinds", nargs="+", default=["advantage", "advantage_fixed", "gbm_cls"], choices=KINDS)
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiment2/ablations"),
                        help="Directory to save ablation evaluation results")
    parser.add_argument("--use_features_parquet", action="store_true",
                        help="Read features from results/features/{dataset}.parquet (Contract 2) instead of the cache")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--config_dir", type=Path, default=Path("configs"))
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    return parser.parse_args()


def write_tables(summary: dict, importances: pd.DataFrame, path: Path, groups, kinds, datasets) -> None:
    lines = ["# Feature ablations: held-out routed nDCG@10 minus always-hybrid",
             "",
             "5-fold CV within each dataset (Experiment 2 folds); alpha*, targets, theta and models fitted on "
             "training folds only. Cell = mean paired difference [95% bootstrap CI]; negative = worse than "
             "always-hybrid. 'Hybrid' is the always-hybrid mean nDCG@10 the differences are measured against.", ""]
    for kind in kinds:
        lines += [f"## Router: `{kind}`", "",
                  "| Features (n) | " + " | ".join([*datasets, "**pooled**"]) + " |",
                  "| :--- | " + " | ".join(":---:" for _ in range(len(datasets) + 1)) + " |"]
        for g in groups:
            row = [f"{g} ({len(FEATURE_GROUPS[g])})"]
            for key in [*datasets, "pooled"]:
                t = summary[kind][g][key]["tests"]["routed - hybrid"]
                row.append(fmt_ci(t))
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "Route shares (BM25 / dense / hybrid), pooled: " + "; ".join(
            f"{g}: " + "/".join(f"{summary[kind][g]['pooled']['route_share'][r]:.0%}" for r in ('bm25', 'dense', 'hybrid'))
            for g in groups), ""]
    base = summary[kinds[0]][groups[0]]
    lines += ["## Always-hybrid mean nDCG@10", "",
              "| " + " | ".join([*datasets, "pooled"]) + " |", "| " + " | ".join(":---:" for _ in range(len(datasets) + 1)) + " |",
              "| " + " | ".join(f"{base[k]['mean']['hybrid']:.4f}" for k in [*datasets, "pooled"]) + " |", ""]
    if not importances.empty:
        top = importances.groupby("feature")[["delta_bm25", "delta_dense"]].mean()
        top["mean"] = top.mean(axis=1)
        top = top.sort_values("mean", ascending=False)
        lines += ["## Gain importance of the advantage regressors (all features, mean over datasets)", "",
                  "| Feature | Delta_bm25 regressor | Delta_dense regressor |", "| :--- | :---: | :---: |"]
        lines += [f"| {f} | {r.delta_bm25:.3f} | {r.delta_dense:.3f} |" for f, r in top.iterrows()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")
    router_cfg = load_config(args.config_dir / "router.yaml")
    seed, k_folds = exp_cfg["seed"], exp_cfg["k_folds"]
    seed_everything(seed)
    alphas = [float(a) for a in hyb_cfg["alpha_grid"]]
    tables = {ds: load_route_table(ds, args.results_dir, args.use_features_parquet) for ds in args.datasets}

    summary, per_query, imp_rows = {}, [], []
    for kind in args.kinds:
        summary[kind] = {}
        for group in args.groups:
            cols = FEATURE_GROUPS[group]
            held_all = []
            summary[kind][group] = {}
            for ds, table in tables.items():
                held, _ = cv_predict(table, kind, cols, alphas, router_cfg, k_folds, seed)
                summary[kind][group][ds] = summarize_heldout(held, args.n_bootstrap, seed)
                held_all.append(prefix_index(held, ds).assign(dataset=ds, kind=kind, group=group))
            pooled = pd.concat(held_all)
            summary[kind][group]["pooled"] = summarize_heldout(pooled, args.n_bootstrap, seed)
            per_query.append(pooled[["dataset", "kind", "group", "fold", "hybrid", "routed", "route_predicted"]])
            t = summary[kind][group]["pooled"]["tests"]["routed - hybrid"]
            log.info(f"[{kind}/{group:<16s}] pooled routed-hybrid = {fmt_ci(t)}")

    if any(k.startswith("advantage") for k in args.kinds):
        for ds, table in tables.items():
            alpha = select_alpha(table, alphas)
            tgt = advantage_targets(table, alpha)
            final = AdvantageRouter(**advantage_kwargs(router_cfg, seed)).fit(
                table[FEATURE_GROUPS["all"]], tgt["delta_bm25"], tgt["delta_dense"], calibrate=False)
            imp = final.feature_importances()
            if imp is not None:
                imp.index.name = "feature"
                imp_rows.append(imp.reset_index().assign(dataset=ds))
    importances = pd.concat(imp_rows) if imp_rows else pd.DataFrame()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(per_query).to_csv(args.output_dir / "per_query_heldout.csv")
    if not importances.empty:
        importances.to_csv(args.output_dir / "feature_importances.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    write_tables(summary, importances, args.output_dir / "tables.md", args.groups, args.kinds, list(tables))
    log.info(f"Wrote {args.output_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
