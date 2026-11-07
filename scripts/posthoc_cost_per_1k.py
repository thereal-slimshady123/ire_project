"""POST HOC (not pre-registered): cost per 1,000 queries and the cost-quality frontier of every Phase 1 policy.

No retrieval is re-run. Inputs (all saved):
  results/policy_latency/policy_latency.csv     held-out nDCG@10 + measured per-query latency of each policy
  results/policy_latency/{ds}_components.csv    per-query component timings (dense-only latency is composed from these)
  results/experiment2/cache/{ds}.csv            per-query dense nDCG@10 (dense needs no tuning, so no CV is involved)

Cost is reported as compute time: X ms per query = X seconds per 1,000 queries (one query at a time, the setup of
report section 3.5: RTX 2050 for the dense encoder and cross-encoder, CPU for BM25 / FAISS / fusion / features). Money
is not reported because it needs an hourly hardware price, which is a choice, not a measurement:
cost per 1,000 queries = (seconds per 1,000 queries / 3600) x price per hour.

Frontier: the deployable policies (oracle selection excluded, it needs relevance labels) that no other policy beats on
both cost and quality, per dataset.

Usage:
  python scripts/posthoc_cost_per_1k.py [--output_dir results/cost_per_1k_posthoc] [--figure report/figures/cost_quality_frontier.png]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["nfcorpus", "scifact", "fiqa"]
GPU_PARTS = ["dense_encode", "cross_encoder_rerank"]

# family -> (label, colour slot, marker); colours: reference categorical palette slots 1-5 (validated, light mode)
FAMILIES = {
    "single": ("Single retriever", "#2a78d6", "o"),
    "hybrid": ("Fixed hybrid", "#eb6834", "s"),
    "router": ("Router / skip-dense", "#1baf7a", "D"),
    "escalation": ("Selective rerank (5–50%)", "#eda100", "^"),
    "rerank": ("Always rerank", "#e87ba4", "v"),
}
NAMES = {"always_bm25": "BM25", "always_dense": "Dense", "always_hybrid": "Hybrid", "gbm_router": "GBM router",
         "skip_dense_gbm": "Skip-dense router", "always_rerank_minilm_l6": "Always rerank (MiniLM-L6)"}


def family(system: str) -> str:
    if system in ("always_bm25", "always_dense"):
        return "single"
    if system == "always_hybrid":
        return "hybrid"
    if system in ("gbm_router", "skip_dense_gbm"):
        return "router"
    if system.startswith("always_rerank"):
        return "rerank"
    return "escalation"


def pretty(system: str) -> str:
    if system in NAMES:
        return NAMES[system]
    base, rate = system.rsplit("_", 1)
    return {"predicted_gain_escalation": "Predicted-gain", "router_confidence_escalation": "Router-confidence",
            "random_escalation": "Random", "router_tau_escalation": "Router + τ"}[base] + f" escalation {rate}"


def load() -> pd.DataFrame:
    pol = pd.read_csv(ROOT / "results/policy_latency/policy_latency.csv")
    pol = pol[~pol["system"].str.startswith("oracle")].copy()  # oracle selection uses relevance labels
    rows = []
    for ds in DATASETS:
        comp = pd.read_csv(ROOT / f"results/policy_latency/{ds}_components.csv", index_col=0)
        cache = pd.read_csv(ROOT / f"results/experiment2/cache/{ds}.csv", index_col=0)
        dense_ms = comp["dense_encode"] + comp["faiss_search"]
        rows.append({"dataset": ds, "system": "always_dense", "ndcg_10": float(cache["ndcg_dense"].mean()),
                     "n": len(cache), "mean_ms": float(dense_ms.mean()), "p50_ms": float(dense_ms.median()),
                     "p95_ms": float(dense_ms.quantile(0.95))})
        # GPU share of the always-on policies (components are additive for them)
        m = comp.drop(columns="dataset").mean()
        parts = {"always_bm25": ["bm25_search"], "always_dense": ["dense_encode", "faiss_search"],
                 "always_hybrid": ["bm25_search", "dense_encode", "faiss_search", "hybrid_fusion"],
                 "always_rerank_minilm_l6": ["bm25_search", "dense_encode", "faiss_search", "hybrid_fusion",
                                             "cross_encoder_rerank"]}
        for sysname, cols in parts.items():
            gpu = sum(m[c] for c in cols if c in GPU_PARTS)
            rows.append({"dataset": ds, "system": sysname, "gpu_share": float(gpu / sum(m[c] for c in cols))})
    extra = pd.DataFrame(rows)
    dense = extra[extra["system"].eq("always_dense") & extra["ndcg_10"].notna()].drop(columns="gpu_share")
    df = pd.concat([pol, dense], ignore_index=True)
    share = extra.dropna(subset=["gpu_share"])[["dataset", "system", "gpu_share"]]
    df = df.merge(share, on=["dataset", "system"], how="left")
    df["family"] = df["system"].map(family)
    df["name"] = df["system"].map(pretty)
    df["seconds_per_1k"] = df["mean_ms"]  # ms per query == seconds per 1,000 queries
    hyb = df[df.system == "always_hybrid"].set_index("dataset")
    df["x_hybrid_cost"] = df["mean_ms"] / df["dataset"].map(hyb["mean_ms"])
    df["ndcg_minus_hybrid"] = df["ndcg_10"] - df["dataset"].map(hyb["ndcg_10"])
    return df


def frontier(g: pd.DataFrame) -> pd.Series:
    """True for policies not dominated (cheaper-or-equal AND better-or-equal, one strictly) by any other."""
    c, q = g["mean_ms"].to_numpy(), g["ndcg_10"].to_numpy()
    dom = [bool(np.any((c <= c[i]) & (q >= q[i]) & ((c < c[i]) | (q > q[i])))) for i in range(len(g))]
    return pd.Series(~np.array(dom), index=g.index)


def plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=surface)
    for ax, ds in zip(axes, DATASETS):
        g = df[df.dataset == ds]
        ax.set_facecolor(surface)
        f = g[g.on_frontier].sort_values("mean_ms")
        ax.step(f["mean_ms"], f["ndcg_10"], where="post", color=ink2, lw=1.2, ls="--", zorder=1)
        for fam, (label, colour, marker) in FAMILIES.items():
            h = g[g.family == fam]
            ax.scatter(h["mean_ms"], h["ndcg_10"], s=46, c=colour, marker=marker, edgecolors=surface,
                       linewidths=1.5, zorder=3, label=label)
        # (dx, dy, ha) in points; left-aligned labels go right of the mark, right-aligned ones to its left
        offsets = {"always_bm25": (6, 6, "left"), "always_dense": (-7, -3, "right"), "always_hybrid": (-7, 4, "right"),
                   "gbm_router": (7, -10, "left"), "always_rerank_minilm_l6": (0, -15, "center"),
                   "skip_dense_gbm": (-7, -11, "right")}
        for sysname, (dx, dy, ha) in offsets.items():
            r = g[g.system == sysname]
            if len(r):
                ax.annotate("Rerank all" if sysname.startswith("always_rerank") else NAMES[sysname], (r.mean_ms.iloc[0], r.ndcg_10.iloc[0]),
                            textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8, color=ink2)
        ax.set_xscale("log")
        ax.set_title(ds, color=ink, fontsize=11, loc="left")
        ax.set_xlabel("Mean latency per query, ms (= seconds per 1,000 queries), log scale", color=ink2, fontsize=8)
        ax.grid(True, color=grid, lw=0.6)
        ax.tick_params(colors=ink2, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Held-out nDCG@10", color=ink2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=ink2, lw=1.2, ls="--"))
    labels.append("Cost–quality frontier")
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8, labelcolor=ink2)
    fig.suptitle("Cost vs quality of every Phase 1 policy (post hoc; RTX 2050, one query at a time)",
                 color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=surface)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output_dir", type=Path, default=ROOT / "results/cost_per_1k_posthoc")
    ap.add_argument("--figure", type=Path, default=ROOT / "report/figures/cost_quality_frontier.png")
    args = ap.parse_args()
    df = load()
    df["on_frontier"] = pd.concat([frontier(g) for _, g in df.groupby("dataset", sort=False)])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "system", "name", "family", "n", "ndcg_10", "ndcg_minus_hybrid", "mean_ms", "p50_ms", "p95_ms",
            "seconds_per_1k", "x_hybrid_cost", "gpu_share", "on_frontier"]
    df[cols].to_csv(out / "cost_per_1k.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "posthoc": True, "unit": "seconds of compute per 1,000 queries (= mean ms per query)",
        "frontier": {ds: g[g.on_frontier].sort_values("mean_ms")["system"].tolist() for ds, g in df.groupby("dataset")},
    }, indent=2), encoding="utf-8")

    key = ["always_bm25", "always_dense", "always_hybrid", "gbm_router", "skip_dense_gbm",
           "predicted_gain_escalation_10%", "router_tau_escalation_10%", "random_escalation_10%", "always_rerank_minilm_l6"]
    L = ["# Cost per 1,000 queries and cost–quality frontier — POST HOC", "",
         "Not pre-registered. Built by `scripts/posthoc_cost_per_1k.py` from saved results only "
         "(`results/policy_latency/`, `results/experiment2/cache/`); no retrieval was re-run. Held-out nDCG@10; latency "
         "measured on an RTX 2050 (dense encoder, cross-encoder) and CPU (BM25, FAISS, fusion, features), one query at a "
         "time (report §3.5). **Cost = compute time: X ms per query = X seconds per 1,000 queries.** To convert to money, "
         "multiply seconds per 1,000 queries / 3,600 by the hourly price of the hardware. Oracle selection is excluded "
         "(it needs relevance labels). Skip-dense was only measured on SciFact.", ""]
    for ds in DATASETS:
        g = df[df.dataset == ds].set_index("system")
        L += [f"## {ds}", "", "| Policy | nDCG@10 | − hybrid | Seconds per 1,000 queries (mean) | p95 ms per query | × hybrid cost | GPU share | On frontier |",
              "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
        for s in key:
            if s in g.index:
                r = g.loc[s]
                gs = "—" if pd.isna(r.gpu_share) else f"{r.gpu_share:.0%}"
                L.append(f"| {r['name']} | {r.ndcg_10:.4f} | {r.ndcg_minus_hybrid:+.4f} | {r.seconds_per_1k:.1f} | "
                         f"{r.p95_ms:.1f} | {r.x_hybrid_cost:.2f} | {gs} | {'yes' if r.on_frontier else ''} |")
        fr = g[g.on_frontier].sort_values("mean_ms")
        L += ["", "Frontier (cheapest first): " + " → ".join(f"{r['name']} ({r.mean_ms:.1f} s, {r.ndcg_10:.4f})"
                                                           for _, r in fr.iterrows()), ""]
    L += ["**Reading the frontier.** Points above hybrid gain at most +0.0034 nDCG@10 and are within noise: Experiment 4 "
          "found no selector significantly better than random escalation (`results/experiment4/tables.md`), and the random "
          "policy here is one seeded sample. The robust conclusions are the large gaps: hybrid costs about the same as "
          "dense alone and is far better; every rerank policy costs 1.6–58× hybrid for no reliable gain.", ""]
    L += ["All budgets (5–50%) and policies: `cost_per_1k.csv`. Figure: `report/figures/cost_quality_frontier.png`."]
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    plot(df, args.figure)
    print("\n".join(L))


if __name__ == "__main__":
    main()
