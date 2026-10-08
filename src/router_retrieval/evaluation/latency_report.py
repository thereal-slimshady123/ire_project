"""Latency benchmarking and aggregation reports."""

from typing import Dict, List, Any
from pathlib import Path
import pandas as pd


def generate_latency_report(
    component_latency_data: Dict[str, List[float]],
) -> pd.DataFrame:
    """
    Generate summary table of latencies by component.

    Computes:
        - Mean (ms)
        - Median / p50 (ms)
        - p95 (ms)
        - p99 (ms)
        - Min / Max (ms)

    Args:
        component_latency_data: Map of component name to list of recorded latencies in ms.

    Returns:
        DataFrame with aggregated statistics per component.
    """
    raise NotImplementedError


def save_latency_report(report_df: pd.DataFrame, output_path: Path) -> None:
    """Save latency report to JSON / CSV format."""
    raise NotImplementedError
