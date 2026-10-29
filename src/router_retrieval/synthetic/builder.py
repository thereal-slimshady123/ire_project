"""Label synthetic queries (BM25 / dense / hybrid-grid nDCG@10 and features) into a route table.

The output has the same schema as the natural route tables (`ndcg_bm25`, `ndcg_dense`, `ndcg_hyb_<alpha>`
and the 18 features), so `router.cv.fit_predict_fold` can concatenate it with natural training queries.
A synthetic query has exactly one relevant document, its source passage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from router_retrieval.data.schema import Candidate, Document, Query
from router_retrieval.evaluation.ranking_metrics import ndcg_at_k
from router_retrieval.features.feature_builder import FEATURE_COLUMNS, FeatureBuilder
from router_retrieval.retrieval.hybrid import BM25_TRUNCATED_FLOOR, fuse_normalized_scores, normalize_for_fusion
from router_retrieval.synthetic.generator import SyntheticQuery

META_COLUMNS = ["dataset", "style", "source_doc", "query_text"]


def build_route_table(
    queries: Sequence[SyntheticQuery],
    bm25,
    dense,
    alphas: Sequence[float],
    corpus: Optional[Dict[str, Document]] = None,
    k: int = 100,
    normalization: str = "min_max",
    feature_builder: Optional[FeatureBuilder] = None,
) -> pd.DataFrame:
    """
    Retrieve with BM25 and dense over the FULL corpus, fuse for every alpha, score against the source
    passage and extract features.

    Args:
        queries: Generated queries with their source passages.
        bm25, dense: Objects exposing `batch_search(queries, k) -> {qid: [Candidate]}`.
        alphas: Hybrid weights (the natural route tables' alpha grid).
        corpus: doc_id -> Document; needed for the query-term-coverage features.
        feature_builder: Defaults to `FeatureBuilder.from_bm25(bm25, corpus)`.
    """
    qobjs = [Query(q.query_id, q.text) for q in queries]
    bm25_run = bm25.batch_search(qobjs, k)
    dense_run = dense.batch_search(qobjs, k)
    for q in qobjs:
        bm25_run.setdefault(q.query_id, [])
        dense_run.setdefault(q.query_id, [])

    builder = feature_builder or FeatureBuilder.from_bm25(bm25, corpus)
    feats = builder.build_feature_matrix(qobjs, bm25_run, dense_run)

    def ids(cands: List[Candidate]) -> List[str]:
        return [c.doc_id for c in sorted(cands, key=lambda c: c.rank)]

    rows = []
    for q in queries:
        rel = {q.source_doc: 1}
        b, d = bm25_run[q.query_id], dense_run[q.query_id]
        nb = normalize_for_fusion(b, k, normalization, BM25_TRUNCATED_FLOOR)
        nd = normalize_for_fusion(d, k, normalization)
        row = {"query_id": q.query_id, "dataset": q.dataset, "ndcg_bm25": ndcg_at_k(ids(b), rel, 10),
               "ndcg_dense": ndcg_at_k(ids(d), rel, 10)}
        for a in alphas:
            fused = fuse_normalized_scores(q.query_id, nb, nd, float(a), k)
            row[f"ndcg_hyb_{float(a)}"] = ndcg_at_k(ids(fused), rel, 10)
        row.update(style=q.style, source_doc=q.source_doc, query_text=q.text)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("query_id")
    table = table.join(feats)
    ordered = ["dataset", "ndcg_bm25", "ndcg_dense", *[c for c in table.columns if c.startswith("ndcg_hyb_")],
               *FEATURE_COLUMNS, "style", "source_doc", "query_text"]
    return table[ordered]


# ------------------------------------------------------------------------------ persistence

def save_synthetic(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.rename_axis("query_id").reset_index()
    out["query_id"] = out["query_id"].astype(str)
    out.to_parquet(path, index=False)


def load_synthetic(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["query_id"] = df["query_id"].astype(str)
    return df.set_index("query_id")


def subsample_synthetic(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Style-balanced subsample of `n` rows with nested subsets: within each style rows are ordered by a
    fixed seeded permutation and the first n/2 (per style) are taken, so n=1000 is contained in n=3000.
    """
    if n <= 0 or df.empty:
        return df.iloc[0:0]
    n = min(n, len(df))
    rng = np.random.default_rng(seed)
    perm = pd.Series(rng.permutation(len(df)), index=df.index)
    ordered = df.assign(_rank=perm).sort_values("_rank")
    styles = sorted(ordered["style"].unique())
    quota = {s: n // len(styles) + (1 if i < n % len(styles) else 0) for i, s in enumerate(styles)}
    parts = [ordered[ordered["style"] == s].head(quota[s]) for s in styles]
    picked = pd.concat(parts)
    short = n - len(picked)  # a style ran out: top up from the remaining rows
    if short > 0:
        picked = pd.concat([picked, ordered.drop(picked.index).head(short)])
    return picked.sort_values("_rank").drop(columns="_rank")
