"""Train and cross-validate the query routers; write Contract 3 held-out predictions.

For every dataset: 5-fold CV over queries (same folds as Experiment 2). Within each training fold the
hybrid weight alpha*, the advantage targets, the risk thresholds theta and the probability temperature are
fitted on training queries only. Outputs per dataset (default `results/models/{dataset}/`):
  predictions.parquet        Contract 3 (kind=advantage; other kinds -> predictions_{kind}.parquet)
  summary.json               held-out means + paired tests vs always-hybrid
  router_{kind}.pkl          final model fitted on all of the dataset's queries (inspection / reuse)
  feature_importances.csv    gain importances of the final advantage regressors

Usage:
  python scripts/train_router.py --datasets scifact fiqa
  python scripts/train_router.py --kind all --stability
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
    FEATURE_GROUPS, KINDS, advantage_kwargs, cv_predict, fmt_ci, load_route_table, predictions_frame,
    summarize_heldout,
)
from router_retrieval.router.gbm_router import AdvantageRouter  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402

DATASETS = ["nfcorpus", "scifact", "fiqa", "cqadupstack-unix", "quora"]
log = get_logger("train_router")


def parse_args():
    parser = argparse.ArgumentParser(description="Train logistic regression, GBM and advantage-regression routers")
    parser.add_argument("--dataset", "--datasets", dest="datasets", nargs="+", default=DATASETS,
                        help="Dataset name(s) (default: all five evaluation datasets)")
    parser.add_argument("--kind", default="advantage", choices=[*KINDS, "all"])
    parser.add_argument("--features", default="all", choices=sorted(FEATURE_GROUPS))
    parser.add_argument("--stability", action="store_true", help="Drop training queries with unstable winners")
    parser.add_argument("--output_dir", type=Path, default=Path("results/models"),
                        help="Directory to save trained model artifacts")
    parser.add_argument("--use_features_parquet", action="store_true",
                        help="Read features from results/features/{dataset}.parquet (Contract 2) instead of the cache")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--config_dir", type=Path, default=Path("configs"))
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")
    router_cfg = load_config(args.config_dir / "router.yaml")
    seed, k_folds = exp_cfg["seed"], exp_cfg["k_folds"]
    seed_everything(seed)
    alphas = [float(a) for a in hyb_cfg["alpha_grid"]]
    kinds = list(KINDS) if args.kind == "all" else [args.kind]
    cols = FEATURE_GROUPS[args.features]

    rows = []
    for ds in args.datasets:
        table = load_route_table(ds, args.results_dir, args.use_features_parquet)
        out_dir = args.output_dir / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {"dataset": ds, "features": args.features, "n_features": len(cols), "kinds": {}}
        for kind in kinds:
            held, infos = cv_predict(table, kind, cols, alphas, router_cfg, k_folds, seed,
                                     stability=True if (args.stability and kind == "advantage") else None)
            name = "predictions.parquet" if kind == "advantage" else f"predictions_{kind}.parquet"
            predictions_frame(held).to_parquet(out_dir / name, index=False)
            s = summarize_heldout(held, args.n_bootstrap, seed)
            summary["kinds"][kind] = {**s, "folds": [{k: v for k, v in i.items() if k != "importances"} for i in infos]}
            t = s["tests"]["routed - hybrid"]
            log.info(f"[{ds}/{kind}] hybrid={s['mean']['hybrid']:.4f} routed={s['mean']['routed']:.4f} "
                     f"routed-hybrid={fmt_ci(t)} shares="
                     + "/".join(f"{s['route_share'][r]:.0%}" for r in ("bm25", "dense", "hybrid")))
            rows.append({"dataset": ds, "kind": kind, **{f"mean_{k}": v for k, v in s["mean"].items()},
                         "routed_minus_hybrid": t["delta_mean"], "ci_low": t["ci_95_low"], "ci_high": t["ci_95_high"]})

            # Final model on all queries of the dataset (for inspection / deployment); alpha* from all of them.
            if kind == "advantage":
                alpha = select_alpha(table, alphas)
                tgt = advantage_targets(table, alpha)
                final = AdvantageRouter(**advantage_kwargs(router_cfg, seed)).fit(
                    table[cols], tgt["delta_bm25"], tgt["delta_dense"])
                final.save(out_dir / "router_advantage.pkl")
                imp = final.feature_importances()
                if imp is not None:
                    imp.sort_values("delta_bm25", ascending=False).to_csv(out_dir / "feature_importances.csv")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    pd.DataFrame(rows).to_csv(args.output_dir / "cv_summary.csv", index=False)
    log.info(f"Wrote {args.output_dir / 'cv_summary.csv'}")


if __name__ == "__main__":
    main()
