"""Information retrieval ranking metrics evaluation (pytrec_eval wrapper)."""

from typing import Dict, List, Sequence
import pandas as pd
from router_retrieval.data.schema import Candidate, QRel


def compute_ranking_metrics(
    qrels: List[QRel],
    run: Dict[str, List[Candidate]],
    metrics: Sequence[str] = ("ndcg_cut_10", "recip_rank@10", "map_cut_100", "recall_100"),
) -> pd.DataFrame:
    """
    Compute ranking metrics per query using pytrec_eval (pytrec-eval-terrier package).

    - 'recip_rank@10' (MRR@10) = pytrec_eval 'recip_rank' on the run truncated to the top 10.
    - The run is converted with score = 1 / rank, because pytrec_eval re-sorts by score and
      would otherwise scramble lists that mix score scales (e.g. escalated runs).
    - Queries in qrels but missing from the run score 0.0 on every metric.

    Args:
        qrels: List of QRel ground truth relevance annotations.
        run: Mapping of query_id to ranked list of Candidates (depth 100).
        metrics: Metric names (see configs/experiment.yaml).

    Returns:
        DataFrame with one row per query: query_id and one column per metric.
        Use aggregate_metrics for means.
    """
    raise NotImplementedError


def aggregate_metrics(metrics_df: pd.DataFrame) -> Dict[str, float]:
    """Compute mean metric values across queries."""
    raise NotImplementedError
