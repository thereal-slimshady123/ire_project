"""Advantage targets and label-stability filtering for the Dual Advantage Regression router.

All functions operate on a "route table": one row per query, with per-route nDCG@10 columns
`ndcg_bm25`, `ndcg_dense`, `ndcg_hyb_<alpha>` (one per hybrid weight on the alpha grid) next to the
feature columns.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from router_retrieval.router.gbm_router import route_label


def hybrid_column(alpha: float) -> str:
    return f"ndcg_hyb_{float(alpha)}"


def alphas_in_table(table: pd.DataFrame) -> List[float]:
    """Hybrid weights present in a route table, ascending."""
    prefix = "ndcg_hyb_"
    return sorted(float(c[len(prefix):]) for c in table.columns if c.startswith(prefix))


def select_alpha(table: pd.DataFrame, alphas: Optional[Sequence[float]] = None) -> float:
    """alpha* = the grid value with the highest mean hybrid nDCG@10 (ties -> smallest alpha)."""
    alphas = list(alphas) if alphas is not None else alphas_in_table(table)
    return max(alphas, key=lambda a: table[hybrid_column(a)].mean())


def advantage_targets(table: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Delta_bm25, Delta_dense (nDCG@10 minus the Hybrid_alpha nDCG@10) and the hybrid score itself."""
    hyb = table[hybrid_column(alpha)]
    return pd.DataFrame(
        {"delta_bm25": table["ndcg_bm25"] - hyb, "delta_dense": table["ndcg_dense"] - hyb, "ndcg_hybrid": hyb},
        index=table.index,
    )


def winner_labels(table: pd.DataFrame, alpha: float, epsilon: float = 0.02) -> np.ndarray:
    """Best route per query against Hybrid_alpha with margin epsilon (near-ties -> hybrid)."""
    t = advantage_targets(table, alpha)
    return route_label(t["delta_bm25"].to_numpy(), t["delta_dense"].to_numpy(), epsilon)


def stability_mask(
    table: pd.DataFrame,
    alpha: float,
    alphas: Optional[Sequence[float]] = None,
    shift: float = 0.1,
    epsilon: float = 0.02,
) -> pd.Series:
    """
    Label-robustness filter (improvements.md section 2.3): keep queries whose winning route does not flip
    when the hybrid weight moves from alpha* to alpha* +/- shift. Neighbours outside the grid are skipped.
    """
    grid = list(alphas) if alphas is not None else alphas_in_table(table)
    neighbours = [a for a in grid if abs(abs(a - alpha) - shift) < 1e-6]
    base = winner_labels(table, alpha, epsilon)
    keep = np.ones(len(table), dtype=bool)
    for a in neighbours:
        keep &= winner_labels(table, a, epsilon) == base
    return pd.Series(keep, index=table.index, name="stable")


def label_noise_summary(table: pd.DataFrame, alpha: float, alphas: Optional[Sequence[float]] = None,
                        shift: float = 0.1, epsilon: float = 0.02) -> Dict[str, float]:
    """Share of queries whose winner is unstable, and the route mix before / after filtering."""
    keep = stability_mask(table, alpha, alphas, shift, epsilon).to_numpy()
    labels = winner_labels(table, alpha, epsilon)
    mix = lambda lab: {r: float((lab == r).mean()) if len(lab) else 0.0 for r in ("bm25", "dense", "hybrid")}
    return {"unstable_share": float(1.0 - keep.mean()) if len(keep) else 0.0,
            "mix_all": mix(labels), "mix_stable": mix(labels[keep])}
