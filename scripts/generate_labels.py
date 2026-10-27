"""Generate descriptive per-query route labels and regret weights.

Default source: Experiment 2's per-query table (results/experiment2/cache/{dataset}.csv) for all five
datasets. Each query is labelled with the hybrid alpha* and margin epsilon of the training fold it was
held out from (results/experiment2/summary.json), i.e. the labels the routers were scored against in the
confusion matrices. These are outcome-derived targets: routers are trained on labels recomputed inside each
training fold, never on these held-out labels.

`--scores` instead labels a single wide CSV with ndcg_bm25 / ndcg_dense / ndcg_hybrid columns (e.g. an
Experiment 1 per_query.csv) at one fixed epsilon.

Usage:
  python scripts/generate_labels.py
  python scripts/generate_labels.py --dataset nfcorpus --scores results/experiment1/nfcorpus/per_query.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.labeling.margin_labeler import MarginLabeler  # noqa: E402
from router_retrieval.labeling.regret import compute_regret_weights  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402

DATASETS = ["nfcorpus", "scifact", "fiqa", "cqadupstack-unix", "quora"]


def label_scores(wide: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """Margin labels + regret columns for a wide (query_id x {bm25, dense, hybrid}) nDCG@10 frame."""
    labels = MarginLabeler(epsilon).label_wide(wide)
    long = wide.rename_axis("query_id").reset_index().melt(id_vars="query_id", var_name="route",
                                                           value_name="ndcg_cut_10")
    regret = compute_regret_weights(long).set_index("query_id")
    out = labels.join(regret)
    for route in wide.columns:
        out[f"ndcg_{route}"] = wide[route]
    out.index.name = "query_id"
    return out


def fold_labels(table: pd.DataFrame, fold_info: list, k_folds: int, seed: int) -> pd.DataFrame:
    """Label every query with the alpha*/epsilon of the training fold it was held out from."""
    frames = []
    for i, (_, test) in enumerate(create_kfold_splits(list(table.index), k_folds, seed)):
        alpha, eps = float(fold_info[i]["alpha"]), float(fold_info[i]["epsilon"])
        wide = pd.DataFrame({"bm25": table.loc[test, "ndcg_bm25"], "dense": table.loc[test, "ndcg_dense"],
                             "hybrid": table.loc[test, f"ndcg_hyb_{alpha}"]})
        out = label_scores(wide, eps)
        out.insert(0, "epsilon", eps)
        out.insert(0, "alpha", alpha)
        out.insert(0, "fold", i)
        frames.append(out)
    return pd.concat(frames).loc[table.index]


def write(result: pd.DataFrame, dataset: str, output_dir: Path) -> Path:
    result.insert(0, "dataset", dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{dataset}_labels.csv"
    result.to_csv(path)
    try:
        result.to_parquet(output_dir / f"{dataset}_labels.parquet")
    except ImportError:
        pass
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--dataset", default=None, help="Dataset name for --scores mode")
    p.add_argument("--scores", type=Path, default=None, help="Wide score CSV with ndcg_bm25/ndcg_dense/ndcg_hybrid")
    p.add_argument("--epsilon", type=float, default=0.02, help="Margin for --scores mode")
    p.add_argument("--exp2_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--output_dir", type=Path, default=Path("results/labels"))
    args = p.parse_args()

    if args.scores:
        if not args.dataset:
            p.error("--scores requires --dataset")
        df = pd.read_csv(args.scores, index_col=0, dtype={"query_id": str})
        df.index = df.index.astype(str)
        cols = {"bm25": "ndcg_bm25", "dense": "ndcg_dense", "hybrid": "ndcg_hybrid"}
        missing = set(cols.values()) - set(df.columns)
        if missing:
            raise ValueError(f"score file lacks required columns: {sorted(missing)}")
        wide = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
        print(f"Wrote {write(label_scores(wide, args.epsilon), args.dataset, args.output_dir)}")
        return

    cfg = load_config(args.config_dir / "experiment.yaml")
    summary = json.loads((args.exp2_dir / "summary.json").read_text(encoding="utf-8"))
    lines = ["# Held-out route labels (training-fold α* and ε)", "",
             "Outcome-derived and descriptive: each held-out query labelled with its training fold's α* and ε "
             "(margin labeler; near-ties → hybrid). Routers are trained on labels recomputed inside training folds.", "",
             "| Dataset | n | BM25 | Dense | Hybrid | Mean max regret |", "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for ds in args.datasets:
        table = pd.read_csv(args.exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        table.index = table.index.astype(str)
        result = fold_labels(table, summary["fold_info"][ds], cfg["k_folds"], cfg["seed"])
        share = result["best_route"].value_counts(normalize=True)
        lines.append(f"| {ds} | {len(result)} | {share.get('bm25', 0):.1%} | {share.get('dense', 0):.1%} "
                     f"| {share.get('hybrid', 0):.1%} | {result['max_regret'].mean():.4f} |")
        print(f"Wrote {write(result, ds, args.output_dir)}")
    (args.output_dir / "label_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
