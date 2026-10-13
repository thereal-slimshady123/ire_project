"""Risk-coverage curve computation across escalation threshold sweeps."""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    regrets: np.ndarray,
    escalation_rates: List[float],
) -> pd.DataFrame:
    """
    Compute risk vs coverage points (selective classification) across escalation rates.
    For each rate, tau is the rate-quantile of the confidences; queries with confidence < tau
    are escalated.

    Coverage: fraction of queries handled by cheap routes without escalation (1 - escalation_rate).
    Risk: mean regret of the router's chosen cheap route on the covered queries, where
          regret = best cheap-route nDCG@10 - chosen-route nDCG@10.

    Args:
        confidences: Max prediction probability per query.
        regrets: Per-query regret of the router's argmax cheap route.
        escalation_rates: Escalation rates to evaluate (configs/escalation.yaml risk_coverage_points).

    Returns:
        DataFrame with columns [escalation_rate, tau, coverage, risk].
    """
    raise NotImplementedError
