"""POST HOC (not pre-registered): recompute the CPU cost table of run_cost_frontier.py without the one query
whose timing was corrupted by the laptop suspending (~57 minutes recorded for a single dense encode).

run_cost_frontier.py is pinned by results/preregistration/phase2_strengthening_20261107.pins.txt and is NOT edited.
This script re-reads its saved per-query timings (timings_<device>.csv) and applies the same latency formulas:
  always_hybrid = bm25 + encode + faiss + fusion
  H1            = bm25 + h1_score + (1 - r) * (encode + faiss + fusion)
  router        = bm25 + skip_features + router_inference + (1 - r) * (encode + faiss + fusion)
  break-even    = per-query decision overhead / (encode + faiss + fusion)
Two summaries are reported: (a) means after dropping rows with any component > --max_ms (the original statistic,
minus the corrupted row), (b) medians over all rows (robustness check). Quality numbers are unaffected (they come
from run_e4_matched_rate.py, not from timings) and are not recomputed here.

Usage:
  python scripts/posthoc_cost_frontier_clean.py --timings results/strengthening/fresh/cost_frontier/timings_cpu.csv \
      --output_dir results/strengthening/fresh/cost_frontier
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

COMPONENTS = ["bm25_search", "dense_encode", "faiss_search", "hybrid_fusion", "skip_feature_extraction",
              "router_inference", "h1_score"]
RATES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]


def summarise(m: pd.Series) -> dict:
    dense_part = m["dense_encode"] + m["faiss_search"] + m["hybrid_fusion"]
    over = {"router": m["skip_feature_extraction"] + m["router_inference"], "H1": m["h1_score"]}
    hybrid = m["bm25_search"] + dense_part
    return {
        "components_ms": {c: float(m[c]) for c in COMPONENTS},
        "hybrid_ms": float(hybrid),
        "break_even_skip_rate": {p: float(o / dense_part) for p, o in over.items()},
        "latency_saving_H1@0.20": float(1 - (m["bm25_search"] + over["H1"] + 0.8 * dense_part) / hybrid),
        "frontier_ms": [{"skip_rate": r, "hybrid": float(hybrid),
                         **{p: float(m["bm25_search"] + o + (1 - r) * dense_part) for p, o in over.items()}}
                        for r in RATES],
    }


def main():
    # The tables contain non-ASCII symbols (−, α); Windows consoles default to cp1252 and would crash on print().
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timings", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--max_ms", type=float, default=60_000.0, help="drop rows with any component above this")
    args = ap.parse_args()
    t = pd.read_csv(args.timings, index_col=0)
    bad = t[(t[COMPONENTS] > args.max_ms).any(axis=1)]
    clean = t.drop(index=bad.index)
    groups = [*clean.groupby("dataset", sort=False), ("pooled", clean)]
    res = {"posthoc": True, "dropped_rows": bad.reset_index().to_dict(orient="records"),
           "mean_excluding_dropped": {ds: summarise(g[COMPONENTS].mean()) for ds, g in groups},
           "median_all_rows": {ds: summarise(g[COMPONENTS].median())
                               for ds, g in [*t.groupby("dataset", sort=False), ("pooled", t)]}}
    stem = args.timings.stem.replace("timings_", "")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_{stem}_clean_posthoc.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    L = [f"# Cost table ({stem}) without the suspended measurement — POST HOC", "",
         f"Recomputed from `{args.timings.as_posix()}` by `scripts/posthoc_cost_frontier_clean.py` (not pre-registered). "
         f"Dropped {len(bad)} row(s) with a component above {args.max_ms:,.0f} ms: "
         + ", ".join(f"{r['dataset']} query {i} ({max(r[c] for c in COMPONENTS) / 60000:.1f} min)" for i, r in bad.iterrows())
         + ". Quality numbers are unchanged; see `tables_" + stem + ".md`.", ""]
    for title, block in [("Means, corrupted row excluded (same statistic as the original table)", res["mean_excluding_dropped"]),
                         ("Medians, all rows (robustness check)", res["median_all_rows"])]:
        L += [f"## {title}", "", "| Dataset | " + " | ".join(COMPONENTS) + " | Hybrid ms | Break-even: router / H1 | H1 saving @20% |",
              "| :--- | " + " | ".join([":---:"] * (len(COMPONENTS) + 3)) + " |"]
        for ds, v in block.items():
            L.append(f"| {ds} | " + " | ".join(f"{v['components_ms'][c]:.2f}" for c in COMPONENTS)
                     + f" | {v['hybrid_ms']:.2f} | {v['break_even_skip_rate']['router']:.1%} / {v['break_even_skip_rate']['H1']:.2%}"
                     f" | {v['latency_saving_H1@0.20']:.1%} |")
        L.append("")
    p = res["mean_excluding_dropped"]["pooled"]["frontier_ms"]
    L += ["## Pooled frontier (means, corrupted row excluded)", "", "| Skip rate | Hybrid ms | Router ms | H1 ms |",
          "| :---: | :---: | :---: | :---: |"]
    L += [f"| {r['skip_rate']:.0%} | {r['hybrid']:.2f} | {r['router']:.2f} | {r['H1']:.2f} |" for r in p]
    (out / f"tables_{stem}_clean_posthoc.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
