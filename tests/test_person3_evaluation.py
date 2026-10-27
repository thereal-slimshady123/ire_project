import numpy as np
import pytest

from router_retrieval.evaluation.latency_report import generate_latency_report
from router_retrieval.evaluation.risk_coverage import compute_risk_coverage_curve
from router_retrieval.evaluation.router_metrics import compute_router_classification_metrics


def test_risk_coverage_uses_confidence_gate_and_empty_coverage():
    out = compute_risk_coverage_curve(np.array([.9, .4]), np.array([0., 1.]), [.5, 1.])
    assert out.coverage.tolist() == [.5, 0.]
    assert out.risk.tolist() == [0., 0.]
    assert out.escalation_rate.tolist() == [.5, 1.]


def test_router_metrics_keep_requested_empty_classes():
    out = compute_router_classification_metrics(["hybrid"], ["hybrid"], ["bm25", "hybrid"])
    assert out["macro_f1"] == .5
    assert out["confusion_matrix"] == [[0, 0], [0, 1]]


def test_latency_report_rejects_invalid_samples():
    report = generate_latency_report({"bm25": [1., 3.]})
    assert report.loc["bm25", "mean_ms"] == 2.
    assert report.loc["bm25", "p95_ms"] == pytest.approx(2.9)
    with pytest.raises(ValueError):
        generate_latency_report({"bad": [float("inf")]})
