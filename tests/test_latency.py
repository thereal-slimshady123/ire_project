"""Unit tests for latency measurement, tracker, and reporting."""

import json
import time
import numpy as np
import pytest

from router_retrieval.pipeline.latency import LatencyTracker, ComponentTimer
from router_retrieval.evaluation.latency_report import (
    generate_latency_report,
    save_latency_report,
    format_latency_markdown_table,
    estimate_component_flops,
)


def test_latency_tracker_and_timer():
    tracker = LatencyTracker()
    with tracker.time_component("test_op"):
        time.sleep(0.005)  # 5ms

    assert "test_op" in tracker.records
    assert len(tracker.records["test_op"]) == 1
    assert tracker.records["test_op"][0] >= 4.0  # at least ~4-5ms

    # Add more synthetic timings
    tracker.record("test_op", 10.0)
    tracker.record("test_op", 20.0)
    tracker.record("test_op", 30.0)

    summary = tracker.get_summary()
    assert "test_op" in summary
    stats = summary["test_op"]
    assert stats["count"] == 4.0
    assert stats["min"] >= 4.0
    assert stats["max"] == 30.0
    assert stats["p50"] <= stats["p95"] <= stats["p99"]

    # Test reset
    tracker.reset()
    assert tracker.records == {}
    assert tracker.get_summary() == {}


def test_generate_latency_report():
    data = {
        "bm25": [1.0, 2.0, 3.0, 4.0, 5.0],
        "dense": [10.0, 20.0, 30.0, 40.0, 50.0],
    }
    df = generate_latency_report(data)
    assert len(df) == 2
    assert "bm25" in df.index
    assert "dense" in df.index
    assert df.loc["bm25", "mean_ms"] == pytest.approx(3.0)
    assert df.loc["dense", "p50_ms"] == pytest.approx(30.0)
    assert df.loc["bm25", "samples"] == 5

    # Test empty input
    empty_df = generate_latency_report({})
    assert len(empty_df) == 0


def test_save_latency_report(tmp_path):
    data = {"bm25": [1.0, 2.0, 3.0]}
    df = generate_latency_report(data)

    json_path = tmp_path / "latency.json"
    save_latency_report(df, json_path)
    assert json_path.exists()
    with open(json_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert "bm25" in loaded
    assert loaded["bm25"]["mean_ms"] == 2.0

    csv_path = tmp_path / "latency.csv"
    save_latency_report(df, csv_path)
    assert csv_path.exists()

    md_path = tmp_path / "latency.md"
    save_latency_report(df, md_path)
    assert md_path.exists()
    assert "bm25" in md_path.read_text(encoding="utf-8")


def test_estimate_component_flops():
    flops_dense = estimate_component_flops("dense_encode", seq_len=32)
    assert flops_dense > 1e8  # 2 * 33M * 32 = 2.11 GFLOPs
    flops_faiss = estimate_component_flops("faiss_search", n_docs=1000, dim=384)
    assert flops_faiss == 2 * 1000 * 384
    flops_ce = estimate_component_flops("cross_encoder", pool_k=50)
    assert flops_ce > 1e9
