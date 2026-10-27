"""Generate descriptive per-query route labels and regret weights from a score table.

The produced labels are outcome-derived targets. For predictive evaluation they must
be generated inside each training fold, never from held-out relevance scores.
"""
import argparse
from pathlib import Path
import pandas as pd
from router_retrieval.labeling.margin_labeler import MarginLabeler
from router_retrieval.labeling.regret import compute_regret_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--scores", type=Path, default=None, help="Wide score CSV with BM25/dense/hybrid columns")
    p.add_argument("--output_dir", type=Path, default=Path("results/labels"))
    p.add_argument("--epsilon", type=float, default=0.02)
    args = p.parse_args()
    source = args.scores or Path("results/experiment1") / args.dataset / "per_query.csv"
    df = pd.read_csv(source, index_col=0, dtype={"query_id": str})
    df.index = df.index.astype(str)
    cols = {"bm25": "ndcg_bm25", "dense": "ndcg_dense", "hybrid": "ndcg_hybrid"}
    missing = set(cols.values()) - set(df.columns)
    if missing:
        raise ValueError(f"score file lacks required columns: {sorted(missing)}")
    wide = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    labels = MarginLabeler(args.epsilon).label_wide(wide)
    labels.index = wide.index
    labels.index.name = "query_id"
    long = wide.rename_axis("query_id").reset_index().melt(id_vars="query_id", var_name="route", value_name="ndcg_cut_10")
    regret = compute_regret_weights(long)
    result = labels.join(regret.set_index("query_id"))
    result.insert(0, "dataset", args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.dataset}_labels.csv"
    result.to_csv(path)
    try:
        result.to_parquet(args.output_dir / f"{args.dataset}_labels.parquet")
    except ImportError:
        pass
    print(f"Wrote outcome-derived labels to {path}. Generate targets inside training folds before model evaluation.")


if __name__ == "__main__":
    main()
