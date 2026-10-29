"""Experiment 2b: Dual Advantage Regression router, optionally trained with synthetic queries at scale.

Held-out protocol identical to Experiment 2 (5-fold CV within each dataset, same folds): alpha*, targets,
theta, temperature and regressors are fitted on training folds only. Optional synthetic queries
(`scripts/generate_synthetic_queries.py`) are appended to the TRAINING side of each fold; held-out
evaluation stays on natural queries. Nothing here overwrites the pre-registered Experiment 2 artifacts.

Primary comparison: pooled held-out nDCG@10 of the advantage router minus always-hybrid (paired bootstrap,
10,000 resamples); success iff the 95% CI lower bound > 0.

Usage:
  python scripts/run_experiment2b_advantage.py                      # natural data only
  python scripts/run_experiment2b_advantage.py --synthetic_dir results/synthetic \
      --n_synthetic 0 1000 3000 10000 30000                         # scaling curve
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.router.cv import (  # noqa: E402
    FEATURE_GROUPS, cv_predict, fmt_ci, load_route_table, prefix_index, summarize_heldout,
)
from router_retrieval.synthetic import load_synthetic, subsample_synthetic  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402

DATASETS = ["nfcorpus", "scifact", "fiqa", "cqadupstack-unix", "quora"]
log = get_logger("experiment2b")


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2b: advantage-regression routing (+ synthetic scaling)")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--features", default="all", choices=sorted(FEATURE_GROUPS))
    p.add_argument("--kind", default="advantage", choices=["advantage", "advantage_fixed"])
    p.add_argument("--synthetic_dir", type=Path, default=None, help="Directory of {dataset}.parquet synthetic tables")
    p.add_argument("--n_synthetic", type=int, nargs="+", default=None,
                   help="Synthetic training queries per fold (scaling sweep). Default: 0 (or the full pool)")
    p.add_argument("--synthetic_scope", choices=["in_domain", "pooled"], default="in_domain",
                   help="in_domain: a dataset's own synthetic queries; pooled: all datasets' synthetic queries")
    p.add_argument("--synthetic_weight", type=float, default=None, help="Override advantage.synthetic_weight")
    p.add_argument("--stability", action="store_true", help="Apply the +/-0.1 alpha label-stability filter")
    p.add_argument("--output_dir", type=Path, default=Path("results/experiment2/advantage"))
    p.add_argument("--use_features_parquet", action="store_true",
                        help="Read features from results/features/{dataset}.parquet (Contract 2) instead of the cache")
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--n_bootstrap", type=int, default=10000)
    return p.parse_args()


def synthetic_pool(args, dataset: str, cache: dict):
    if args.synthetic_dir is None:
        return None
    names = [dataset] if args.synthetic_scope == "in_domain" else sorted(
        p.stem for p in Path(args.synthetic_dir).glob("*.parquet"))
    frames = []
    for n in names:
        path = Path(args.synthetic_dir) / f"{n}.parquet"
        if path.exists():
            cache.setdefault(n, load_synthetic(path))
            frames.append(cache[n])
    return pd.concat(frames) if frames else None


def write_tables(summary: dict, scaling: list, datasets: list, path: Path, prereg: dict) -> None:
    pooled0 = scaling[0]["pooled"]
    prim = pooled0["tests"]["routed - hybrid"]
    lines = [
        f"# Experiment 2b: Dual Advantage Regression router (held-out)",
        "",
        "5-fold CV within each dataset (Experiment 2 folds). alpha*, targets, theta, temperature and regressors "
        "are fitted on training folds only; synthetic queries (if any) join the training side only. Differences "
        "are nDCG@10, router minus always-hybrid, [95% paired bootstrap CI].", "",
        f"**Primary (pooled, n = {pooled0['n']}, {scaling[0]['label']}): advantage router − always-hybrid = "
        f"{fmt_ci(prim)}** → {'SUCCESS' if prim['ci_95_low'] > 0 else 'NOT a significant gain'}.", ""]
    if prereg:
        lines += [f"Reference, pre-registered Experiment 2 (3-class GBM, pooled): {fmt_ci(prereg)}.", ""]
    lines += ["## Scaling with synthetic training queries", "",
              "| Synthetic per dataset per fold (cap; realised counts) | " + " | ".join([*datasets, "**pooled**"]) + " | Pooled route shares B/D/H |",
              "| :--- | " + " | ".join(":---:" for _ in range(len(datasets) + 1)) + " | :---: |"]
    for row in scaling:
        cells = [fmt_ci(row[d]["tests"]["routed - hybrid"]) for d in datasets]
        cells.append(fmt_ci(row["pooled"]["tests"]["routed - hybrid"]))
        sh = "/".join(f"{row['pooled']['route_share'][r]:.0%}" for r in ("bm25", "dense", "hybrid"))
        lines.append(f"| {row['label']} | " + " | ".join(cells) + f" | {sh} |")
    last = scaling[-1]["pooled"]
    lines += ["", f"## Detail for the largest setting ({scaling[-1]['label']})", "",
              "| Dataset | n | Hybrid | BM25 | Dense | Router | Random (same mix) | Oracle | Win/Loss/Tie |",
              "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for d in [*datasets, "pooled"]:
        s = scaling[-1][d]
        m, w = s["mean"], s["wins_losses_ties"]
        lines.append(f"| {d} | {s['n']} | {m['hybrid']:.4f} | {m['bm25']:.4f} | {m['dense']:.4f} | {m['routed']:.4f} "
                     f"| {m['random_same_mix']:.4f} | {m['oracle']:.4f} | {w['win']}/{w['loss']}/{w['tie']} |")
    t = last["tests"]
    lines += ["", f"Router vs random routing at the same route mix (pooled): {fmt_ci(t['routed - random_same_mix'])}; "
              f"oracle − hybrid: {fmt_ci(t['oracle - hybrid'])}; share of oracle headroom captured: "
              f"{last['headroom_captured']:.1%}."]
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
    tables = {ds: load_route_table(ds, args.results_dir, args.use_features_parquet) for ds in args.datasets}
    sweep = args.n_synthetic if args.n_synthetic is not None else ([0] if args.synthetic_dir is None else [10 ** 9])

    cache, scaling, per_query = {}, [], []
    for n_syn in sweep:
        held_all, used = [], {}
        for ds, table in tables.items():
            pool = synthetic_pool(args, ds, cache)
            syn = subsample_synthetic(pool, n_syn, seed) if (pool is not None and n_syn > 0) else None
            used[ds] = 0 if syn is None else len(syn)
            held, _ = cv_predict(table, args.kind, cols, alphas, router_cfg, k_folds, seed, synthetic=syn,
                                 synthetic_weight=args.synthetic_weight,
                                 stability=True if args.stability else None)
            held_all.append(prefix_index(held, ds).assign(dataset=ds, n_synthetic=used[ds]))
        pooled = pd.concat(held_all)
        # Label the requested setting (a per-dataset cap), not the largest count actually used.
        realised = ", ".join(f"{d} {used[d]:,}" for d in tables)
        row = {"label": "0 (natural only)" if n_syn == 0 else f"{n_syn:,} per dataset (used: {realised})",
               "n_synthetic": used, "pooled": summarize_heldout(pooled, args.n_bootstrap, seed)}
        for ds in tables:
            row[ds] = summarize_heldout(pooled[pooled["dataset"] == ds], args.n_bootstrap, seed)
        scaling.append(row)
        per_query.append(pooled)
        log.info(f"[n_synthetic={row['label']}] pooled routed-hybrid = "
                 f"{fmt_ci(row['pooled']['tests']['routed - hybrid'])}  shares="
                 + "/".join(f"{row['pooled']['route_share'][r]:.0%}" for r in ("bm25", "dense", "hybrid")))

    prereg = {}
    p2 = args.results_dir / "experiment2" / "summary.json"
    if p2.exists():
        prereg = json.loads(p2.read_text(encoding="utf-8"))["pooled"]["tests"]["router_gbm - hybrid"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(per_query).to_csv(args.output_dir / "per_query_heldout.csv")
    (args.output_dir / "summary.json").write_text(
        json.dumps({"kind": args.kind, "features": args.features, "synthetic_scope": args.synthetic_scope, "stability": args.stability,
                    "scaling": scaling, "preregistered_gbm_pooled": prereg}, indent=2, default=float), encoding="utf-8")
    write_tables({}, scaling, list(tables), args.output_dir / "tables.md", prereg)
    log.info(f"Wrote {args.output_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
