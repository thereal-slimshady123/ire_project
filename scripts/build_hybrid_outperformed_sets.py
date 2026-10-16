"""Build query sets on which a tuned hybrid is outperformed by BM25 or by dense retrieval.

Purpose: an existence demonstration that "there are queries where hybrid gets outperformed", i.e. that
routing has a point in principle. Queries are SELECTED BY OUTCOME from the natural test queries of the
five datasets (that selection is the method, not a flaw): a query enters `bm25_wins` if
nDCG@10(BM25) - nDCG@10(hybrid) >= --min_gap, and `dense_wins` likewise for dense.

Hybrid gets its best shot: per dataset, alpha is tuned in-sample on ALL of that dataset's queries.
A query is flagged `robust` when the win also holds against hybrid at alpha* - 0.1 and alpha* + 0.1.

Outputs:
  data/processed/hybrid-outperformed/{bm25_wins,dense_wins}/queries.jsonl, qrels/test.tsv  (BEIR format;
      query ids are "<dataset>:<id>", documents live in each query's source corpus)
  results/hybrid_outperformed/manifest.csv   (every selected query with its scores)
  results/hybrid_outperformed/tables.md

Usage:
  python scripts/build_hybrid_outperformed_sets.py [--min_gap 0.02]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {
    "nfcorpus": Path("data/raw/nfcorpus"),
    "scifact": Path("data/raw/scifact"),
    "fiqa": Path("data/raw/fiqa"),
    "cqadupstack-unix": Path("data/raw/cqadupstack-unix"),
    "quora": Path("data/processed/quora-sub"),
}
ALPHAS = [round(a, 1) for a in np.arange(0, 1.01, 0.1)]


def parse_args():
    p = argparse.ArgumentParser(description="Build query sets where tuned hybrid is outperformed")
    p.add_argument("--min_gap", type=float, default=0.02, help="Minimum nDCG@10 margin over hybrid")
    p.add_argument("--exp2_cache", type=Path, default=Path("results/experiment2/cache"))
    p.add_argument("--data_out", type=Path, default=Path("data/processed/hybrid-outperformed"))
    p.add_argument("--results_out", type=Path, default=Path("results/hybrid_outperformed"))
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for ds, path in DATASETS.items():
        df = pd.read_csv(args.exp2_cache / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        df.index = df.index.astype(str)
        alpha = max(ALPHAS, key=lambda a: df[f"ndcg_hyb_{a}"].mean())  # in-sample: best case for hybrid
        neighbours = [a for a in (round(alpha - 0.1, 1), round(alpha + 0.1, 1)) if a in ALPHAS]
        texts = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(path / "queries.jsonl", encoding="utf-8")}
        qrels = pd.read_csv(path / "qrels" / "test.tsv", sep="\t", dtype=str)
        for q, r in df.iterrows():
            h = r[f"ndcg_hyb_{alpha}"]
            nb = [r[f"ndcg_hyb_{a}"] for a in neighbours]
            for route in ("bm25", "dense"):
                gap = r[f"ndcg_{route}"] - h
                if gap >= args.min_gap:
                    rows.append({
                        "set": f"{route}_wins", "dataset": ds, "query_id": f"{ds}:{q}", "text": texts[q],
                        "alpha_star": alpha, "ndcg_bm25": r["ndcg_bm25"], "ndcg_dense": r["ndcg_dense"],
                        "ndcg_hybrid": h, "gap": gap,
                        "robust": bool(all(r[f"ndcg_{route}"] - x >= args.min_gap for x in nb)),
                        "_qrels": qrels[qrels["query-id"] == q],
                    })
    sel = pd.DataFrame(rows)
    totals = {ds: len(pd.read_csv(args.exp2_cache / f"{ds}.csv", usecols=[0])) for ds in DATASETS}

    for name, g in sel.groupby("set"):
        out = args.data_out / name
        (out / "qrels").mkdir(parents=True, exist_ok=True)
        with open(out / "queries.jsonl", "w", encoding="utf-8") as f:
            for _, r in g.iterrows():
                f.write(json.dumps({"_id": r.query_id, "text": r.text,
                                    "metadata": {"source_dataset": r.dataset, "selected_because": name}}) + "\n")
        with open(out / "qrels" / "test.tsv", "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for _, r in g.iterrows():
                for _, qr in r["_qrels"].iterrows():
                    f.write(f"{r.query_id}\t{qr['corpus-id']}\t{qr['score']}\n")

    args.results_out.mkdir(parents=True, exist_ok=True)
    sel.drop(columns="_qrels").to_csv(args.results_out / "manifest.csv", index=False)

    lines = [
        "# Queries where a tuned hybrid is outperformed",
        "",
        f"Existence demonstration. Natural test queries from five BEIR/MTEB datasets, **selected because** BM25 or "
        f"dense beat hybrid by ≥ {args.min_gap} nDCG@10. Hybrid α is tuned in-sample on each full dataset (best case "
        "for hybrid). *Robust* = the win also holds against hybrid at α* ± 0.1. Retrievers: BM25 (k1 1.2, b 0.75); "
        "dense `intfloat/e5-small-v2`.",
        "",
        "## Mean nDCG@10 on each selected set",
        "",
        "| Set | Queries | BM25 | Dense | Hybrid (tuned α) | Robust queries |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, g in sel.groupby("set"):
        rob = g[g.robust]
        lines.append(f"| {name} (all) | {len(g)} | {g.ndcg_bm25.mean():.3f} | {g.ndcg_dense.mean():.3f} "
                     f"| {g.ndcg_hybrid.mean():.3f} | {len(rob)} |")
        lines.append(f"| {name} (robust only) | {len(rob)} | {rob.ndcg_bm25.mean():.3f} | {rob.ndcg_dense.mean():.3f} "
                     f"| {rob.ndcg_hybrid.mean():.3f} | — |")
    lines += ["", "## Per source dataset", "",
              "| Dataset | Test queries | BM25 wins (robust) | Dense wins (robust) |", "| :--- | :---: | :---: | :---: |"]
    for ds in DATASETS:
        b = sel[(sel.dataset == ds) & (sel.set == "bm25_wins")]
        d = sel[(sel.dataset == ds) & (sel.set == "dense_wins")]
        lines.append(f"| {ds} | {totals[ds]} | {len(b)} ({b.robust.sum()}) | {len(d)} ({d.robust.sum()}) |")
    for name, g in sel.groupby("set"):
        lines += ["", f"## Largest margins: {name}", "", "| Dataset | Query | Winner | Hybrid | Robust |",
                  "| :--- | :--- | :---: | :---: | :---: |"]
        col = "ndcg_bm25" if name == "bm25_wins" else "ndcg_dense"
        for _, r in g.sort_values("gap", ascending=False).head(10).iterrows():
            lines.append(f"| {r.dataset} | {r.text[:90].replace('|', '/')} | {r[col]:.2f} | {r.ndcg_hybrid:.2f} "
                         f"| {'yes' if r.robust else 'no'} |")
    (args.results_out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = sel.groupby("set").size().to_dict()
    print(f"Wrote {args.data_out} and {args.results_out / 'tables.md'}: {counts}")


if __name__ == "__main__":
    main()
