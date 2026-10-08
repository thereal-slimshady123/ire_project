"""Evaluation metrics, significance testing, and performance reporting."""

from router_retrieval.evaluation.ranking_metrics import compute_ranking_metrics
from router_retrieval.evaluation.router_metrics import compute_router_classification_metrics
from router_retrieval.evaluation.significance import paired_bootstrap_test
from router_retrieval.evaluation.risk_coverage import compute_risk_coverage_curve
from router_retrieval.evaluation.latency_report import generate_latency_report

__all__ = [
    "compute_ranking_metrics",
    "compute_router_classification_metrics",
    "paired_bootstrap_test",
    "compute_risk_coverage_curve",
    "generate_latency_report",
]
