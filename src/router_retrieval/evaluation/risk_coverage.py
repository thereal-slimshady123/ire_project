"""Risk-coverage curve computation across escalation threshold sweeps."""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    errors_or_regrets: np.ndarray,
    taus: List[float],
) -> pd.DataFrame:
    """
    Compute risk vs coverage points across confidence thresholds.

    Coverage: fraction of queries handled by cheap routes without escalation.
    Risk: average regret or error on the covered queries.

    Args:
        confidences: Max prediction probability per query.
        errors_or_regrets: Per-query regret or error of the cheap route.
        taus: List of threshold values swept.

    Returns:
        DataFrame with columns [tau, coverage, risk, escalation_rate].
    """
    raise NotImplementedError
