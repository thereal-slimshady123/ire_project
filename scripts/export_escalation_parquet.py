"""Write Contract 4 (`results/experiment4/{dataset}_escalation.parquet`) from Experiment 4's held-out records.

One row per (query, budget, policy): `query_id`, `budget`, `policy`, `escalated`, `ndcg_final`, `latency_ms`.
`escalated` is re-derived with Experiment 4's rule (within each dataset and fold, escalate the top
round(B * n) queries by the policy's score, stable order), and `ndcg_final` is checked against the stored
per-policy column. The random policy is stored as Experiment 4 computes it, an expected value, so its
`escalated` is null and its latency is the expected B-weighted cost.

`latency_ms` is a per-query estimate from the measured component timings in
`results/policy_latency/{dataset}_components.csv` (hybrid path, plus the policy's scoring model, plus the
cross-encoder if escalated); it is null for datasets without measured components (Unix, Quora).

Usage:
  python scripts/export_escalation_parquet.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("export_escalation")
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]  # as in run_experiment4_escalation.py
SCORED = ["predicted_gain", "router_confidence", "oracle"]
HYBRID_PATH = ["bm25_search", "dense_encode", "faiss_search", "hybrid_fusion"]
SCORING_COST = {"predicted_gain": ["feature_extraction", "gain_model_inference"],
                "router_confidence": ["feature_extraction", "router_inference"],
                "oracle": [], "random": []}


def parse_args():
    p = argparse.ArgumentParser(description="Export Contract 4 escalation parquet files")
    p.add_argument("--exp4_dir", type=Path, default=Path("results/experiment4"))
    p.add_argument("--latency_dir", type=Path, default=Path("results/policy_latency"))
    return p.parse_args()


def escalated_mask(sub: pd.DataFrame, policy: str, budget: float) -> np.ndarray:
    n_esc = int(round(budget * len(sub)))
    order = np.argsort(-sub[f"score_{policy}"].values, kind="stable")
    esc = np.zeros(len(sub), dtype=bool)
    esc[order[:n_esc]] = True
    return esc


def export_dataset(df: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    base = comp[HYBRID_PATH].sum(axis=1) if comp is not None else pd.Series(np.nan, index=df.index)
    base = base.reindex(df.index)
    ce = comp["cross_encoder_rerank"].reindex(df.index) if comp is not None else base * np.nan
    rows = []
    for budget in BUDGETS:
        for policy in [*SCORED, "random"]:
            col = f"{policy}@{budget}"
            extra = comp[SCORING_COST[policy]].sum(axis=1).reindex(df.index) if comp is not None else 0.0
            if policy == "random":
                esc = pd.array([pd.NA] * len(df), dtype="boolean")
                latency = base + extra + budget * ce
            else:
                esc = np.zeros(len(df), dtype=bool)
                for _, idx in df.groupby("fold").indices.items():
                    esc[idx] = escalated_mask(df.iloc[idx], policy, budget)
                final = np.where(esc, df["reranked"].values, df["hybrid"].values)
                if not np.allclose(final, df[col].values):
                    raise RuntimeError(f"re-derived escalation does not reproduce {col}")
                latency = base + extra + np.where(esc, ce, 0.0)
                esc = pd.array(esc, dtype="boolean")
            rows.append(pd.DataFrame({
                "query_id": [q.split(":", 1)[1] for q in df.index],
                "budget": np.float32(budget),
                "policy": policy,
                "escalated": esc,
                "ndcg_final": df[col].values.astype(np.float32),
                "latency_ms": np.asarray(latency, dtype=np.float32),
            }))
    return pd.concat(rows, ignore_index=True)


def main():
    args = parse_args()
    heldout = pd.read_csv(args.exp4_dir / "per_query_heldout.csv", index_col=0)
    for ds, df in heldout.groupby("dataset", sort=False):
        comp_path = args.latency_dir / f"{ds}_components.csv"
        comp = pd.read_csv(comp_path, index_col=0) if comp_path.exists() else None
        out = export_dataset(df, comp)
        path = args.exp4_dir / f"{ds}_escalation.parquet"
        out.to_parquet(path, index=False)
        log.info(f"[{ds}] {len(out)} rows -> {path}" + ("" if comp is not None else " (no latency components)"))


if __name__ == "__main__":
    main()
