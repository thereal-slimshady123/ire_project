"""Information retrieval ranking metrics with trec_eval semantics.

Implemented directly in numpy (gain = graded relevance, log2 discount, ideal DCG over all
judged-relevant documents) and cross-checked against pytrec_eval in tests/test_ranking_metrics.py.
Every query present in the qrels is evaluated; a query with an empty run scores 0.
"""

from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd

from router_retrieval.data.schema import Candidate, QRel

QrelsDict = Dict[str, Dict[str, int]]
Ranking = Union[List[Candidate], List[str]]


def _as_doc_ids(ranking: Ranking) -> List[str]:
    if ranking and isinstance(ranking[0], Candidate):
        return [c.doc_id for c in sorted(ranking, key=lambda c: c.rank)]
    return list(ranking)


def ndcg_at_k(doc_ids: Sequence[str], rels: Dict[str, int], k: int = 10) -> float:
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    if not ideal:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = sum(max(rels.get(d, 0), 0) * discounts[i] for i, d in enumerate(doc_ids[:k]))
    idcg = float(np.dot(ideal, discounts[: len(ideal)]))
    return dcg / idcg


def reciprocal_rank(doc_ids: Sequence[str], rels: Dict[str, int], k: int = None) -> float:
    for i, d in enumerate(doc_ids[:k] if k else doc_ids):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(doc_ids: Sequence[str], rels: Dict[str, int]) -> float:
    n_rel = sum(1 for r in rels.values() if r > 0)
    if n_rel == 0:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(doc_ids):
        if rels.get(d, 0) > 0:
            hits += 1
            total += hits / (i + 1)
    return total / n_rel


def recall_at_k(doc_ids: Sequence[str], rels: Dict[str, int], k: int = 100) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(doc_ids[:k])) / len(relevant)


_METRICS = {
    "ndcg_cut_10": lambda d, r: ndcg_at_k(d, r, 10),
    "recip_rank": lambda d, r: reciprocal_rank(d, r),
    "mrr_10": lambda d, r: reciprocal_rank(d, r, 10),
    "map": average_precision,
    "recall_100": lambda d, r: recall_at_k(d, r, 100),
}


def compute_ranking_metrics(
    qrels: Union[List[QRel], QrelsDict],
    run: Dict[str, Ranking],
    metrics: Sequence[str] = ("ndcg_cut_10", "recip_rank", "map", "recall_100"),
) -> pd.DataFrame:
    """
    Compute ranking metrics per query.

    Args:
        qrels: List of QRel ground truth relevance annotations (or {qid: {doc_id: rel}}).
        run: Mapping of query_id to ranked list of Candidates (or ranked doc_id lists).
        metrics: Metric identifiers; supported: ndcg_cut_10, recip_rank, mrr_10, map, recall_100.

    Returns:
        DataFrame indexed by query_id with one column per metric (use aggregate_metrics for means).
    """
    if isinstance(qrels, list):
        from router_retrieval.data.beir_loader import qrels_to_dict

        qrels = qrels_to_dict(qrels)
    unknown = set(metrics) - _METRICS.keys()
    if unknown:
        raise ValueError(f"Unsupported metrics: {sorted(unknown)}")
    rows = {}
    for qid, rels in qrels.items():
        doc_ids = _as_doc_ids(run.get(qid, []))
        rows[qid] = {m: _METRICS[m](doc_ids, rels) for m in metrics}
    df = pd.DataFrame.from_dict(rows, orient="index", columns=list(metrics))
    df.index.name = "query_id"
    return df


def aggregate_metrics(metrics_df: pd.DataFrame) -> Dict[str, float]:
    """Compute mean metric values across queries."""
    return {col: float(metrics_df[col].mean()) for col in metrics_df.columns}
