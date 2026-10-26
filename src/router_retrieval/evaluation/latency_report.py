import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
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
        - Std (ms)
        - Samples

    Args:
        component_latency_data: Map of component name to list of recorded latencies in ms.

    Returns:
        DataFrame with aggregated statistics per component.
    """
    rows = []
    for component, latencies in component_latency_data.items():
        if not latencies:
            continue
        arr = np.asarray(latencies, dtype=np.float64)
        rows.append(
            {
                "component": component,
                "samples": int(len(arr)),
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["component", "samples", "mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"]
        ).set_index("component")

    df = pd.DataFrame(rows).set_index("component")
    return df


def format_latency_markdown_table(report_df: pd.DataFrame) -> str:
    """Format latency report DataFrame as a GitHub-flavored Markdown table."""
    lines = [
        "| Component | Samples | Mean (ms) | Median / p50 (ms) | p95 (ms) | p99 (ms) | Min (ms) | Max (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for comp, row in report_df.iterrows():
        lines.append(
            f"| `{comp}` | {int(row['samples'])} | {row['mean_ms']:.2f} | {row['p50_ms']:.2f} | "
            f"{row['p95_ms']:.2f} | {row['p99_ms']:.2f} | {row['min_ms']:.2f} | {row['max_ms']:.2f} |"
        )
    return "\n".join(lines)


def estimate_component_flops(
    component: str,
    n_docs: int = 5000,
    dim: int = 384,
    seq_len: int = 32,
    pool_k: int = 50,
    dense_params: int = 33_000_000,  # e5-small-v2 ~ 33M params
    ce_params: int = 22_700_000,     # MiniLM-L6-v2 ~ 22.7M params
) -> float:
    """
    Estimate theoretical floating point operations (FLOPs) for a component.

    - Dense bi-encoder query encode: ~ 2 * N_params * seq_len
    - FAISS flat inner-product search: 2 * n_docs * dim
    - Cross-encoder rerank: ~ 2 * N_params * pool_k * (seq_len + passage_len)
    - BM25 & Router: negligible (~ 1e4 - 1e5 ops)
    """
    comp = component.lower()
    if "dense_encode" in comp or "query_encode" in comp:
        return float(2 * dense_params * seq_len)
    if "faiss" in comp or "vector_search" in comp:
        return float(2 * n_docs * dim)
    if "cross_encoder" in comp or "rerank" in comp:
        # Cross-encoder evaluates (query, doc) pairs of ~160 tokens
        pair_tokens = seq_len + 128
        return float(2 * ce_params * pool_k * pair_tokens)
    if "bm25" in comp:
        return float(n_docs * 50)  # rough inverted-index posting operations
    if "router" in comp:
        return 100_000.0
    return 0.0


def save_latency_report(report_df: pd.DataFrame, output_path: Path) -> None:
    """Save latency report to JSON and CSV formats."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".json":
        # Save structured JSON
        data = {
            comp: {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in row.items()}
            for comp, row in report_df.iterrows()
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif output_path.suffix == ".csv":
        report_df.to_csv(output_path)
    elif output_path.suffix == ".md":
        output_path.write_text(format_latency_markdown_table(report_df) + "\n", encoding="utf-8")
    else:
        # Default: save both csv and json
        report_df.to_csv(output_path.with_suffix(".csv"))
        save_latency_report(report_df, output_path.with_suffix(".json"))

