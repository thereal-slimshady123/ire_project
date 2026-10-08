"""Information retrieval ranking metrics evaluation (pytrec_eval / ir_measures wrapper)."""

from typing import Dict, List, Sequence
import pandas as pd
from router_retrieval.data.schema import Candidate, QRel


def compute_ranking_metrics(
    qrels: List[QRel],
    run: Dict[str, List[Candidate]],
    metrics: Sequence[str] = ("ndcg_cut_10", "recip_rank", "map", "recall_100"),
) -> pd.DataFrame:
    """
    Compute official ranking metrics per query using pytrec_eval.

    Args:
        qrels: List of QRel ground truth relevance annotations.
        run: Mapping of query_id to ranked list of Candidates.
        metrics: Tuple of standard TREC metric identifiers.

    Returns:
        DataFrame containing query_id and metric columns, with an overall mean summary row.
    """
    raise NotImplementedError


def aggregate_metrics(metrics_df: pd.DataFrame) -> Dict[str, float]:
    """Compute mean metric values across queries."""
    raise NotImplementedError
