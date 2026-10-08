"""Context-manager based component latency timers and tracker."""

import time
from typing import Dict, List, Optional
from contextlib import contextmanager


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
        if component_name not in self.records:
            self.records[component_name] = []
        self.records[component_name].append(latency_ms)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute mean, median, p95, p99 latency per component.

        Returns:
            Dictionary mapping component to summary statistics.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset recorded latencies."""
        self.records.clear()
