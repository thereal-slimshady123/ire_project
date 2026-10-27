"""Context-manager based component latency timers and tracker."""

import time
import math
from typing import Dict, List, Optional
from contextlib import contextmanager
import numpy as np


class ComponentTimer:
    """Context manager for timing individual pipeline components."""

    def __init__(self, component_name: str, tracker: "LatencyTracker"):
        self.component_name = component_name
        self.tracker = tracker
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        self.tracker.record(self.component_name, elapsed_ms)


class LatencyTracker:
    """Tracks latency recordings across components and queries."""

    def __init__(self):
        self.records: Dict[str, List[float]] = {}

    def time_component(self, component_name: str) -> ComponentTimer:
        """Create a ComponentTimer context manager."""
        return ComponentTimer(component_name, self)

    def record(self, component_name: str, latency_ms: float) -> None:
        """Record latency measurement for component."""
        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        if component_name not in self.records:
            self.records[component_name] = []
        self.records[component_name].append(latency_ms)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute mean, median, p95, p99 latency per component.

        Returns:
            Dictionary mapping component to summary statistics.
        """
        summary: Dict[str, Dict[str, float]] = {}
        for comp, vals in self.records.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            summary[comp] = {
                "count": float(len(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        return summary

    def reset(self) -> None:
        """Reset recorded latencies."""
        self.records.clear()
