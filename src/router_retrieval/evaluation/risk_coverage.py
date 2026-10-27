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
    conf = np.asarray(confidences, dtype=float)
    risk = np.asarray(errors_or_regrets, dtype=float)
    if conf.ndim != 1 or risk.shape != conf.shape:
        raise ValueError("confidences and errors_or_regrets must be equal-length 1-D arrays")
    if not np.isfinite(conf).all() or not np.isfinite(risk).all():
        raise ValueError("inputs must be finite")
    rows = []
    for tau in taus:
        covered = conf >= tau
        coverage = float(covered.mean()) if len(conf) else 0.0
        rows.append({"tau": float(tau), "coverage": coverage,
                     "risk": float(risk[covered].mean()) if covered.any() else 0.0,
                     "escalation_rate": 1.0 - coverage})
    return pd.DataFrame(rows, columns=["tau", "coverage", "risk", "escalation_rate"])
