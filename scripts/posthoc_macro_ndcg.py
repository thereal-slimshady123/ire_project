"""POST HOC (not pre-registered): category-balanced (macro) nDCG@10 over the E2 query groupings.

The usual (micro) average weights every query equally, so large groups dominate. The macro average gives every group
of a grouping the same weight: macro = mean over groups of the group's mean nDCG@10. Groupings are E2's, computed by
the unchanged `add_groups` of scripts/analyze_niche_existence.py: dataset, query family (exploration only), query
length in words, question form, identifier-like tokens. Groups with fewer than --min_n queries are left out of that
grouping's macro average (their queries still count in the micro average of the grouping's covered queries).

Systems (held-out, no refitting): BM25, dense, hybrid at the per-fold alpha* (Experiment 2d), the cost-aware router
(Experiment 2d), and the per-query oracle (an upper bound, not a policy). Differences use a paired bootstrap that
resamples queries within each group (10,000 resamples, seed 42), so every group keeps its size.

Data:
  validation   CQADupStack mathematica / english / gaming (results/phase2_validation/), held out for E1-E2
  exploration  the 13 Phase 2 exploration datasets (results/experiment2d/, results/niche/experiment2d/); BRIGHT has no
               query text left, so it counts only in the dataset grouping
Query text: data/raw/<ds>/queries.jsonl when present, data/processed/natural-niche/all_queries_with_overlap.csv
(exploration), and data/downloads/queries_only/<ds>/queries.jsonl (query files downloaded from Hugging Face `mteb/<ds>`;
no retrieval).

Usage:
  python scripts/posthoc_macro_ndcg.py [--output_dir results/macro_ndcg_posthoc] [--min_n 30]
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nich", ROOT / "scripts" / "analyze_niche_existence.py")
nich = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nich)

SYSTEMS = {"bm25": "ndcg_bm25", "dense": "ndcg_dense", "hybrid": "hybrid_ndcg_cut_10",
           "router": "cost_aware_ndcg_cut_10", "oracle": "oracle_ndcg_cut_10"}
LABELS = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid (α*)", "router": "Cost-aware router", "oracle": "Oracle (bound)"}
GROUPINGS = {"dataset": "Dataset", "g_family": "Query family", "g_length": "Length (words)", "g_form": "Question form",
             "g_ident": "Identifier-like token"}
SCOPES = {
    "validation": {"pub": ["results/phase2_validation/experiment2d"], "cache": ["results/phase2_validation/experiment2/cache"]},
    "exploration": {"pub": ["results/experiment2d", "results/niche/experiment2d"],
                    "cache": ["results/experiment2/cache", "results/niche/experiment2/cache"]},
}
N_BOOT, SEED = 10000, 42


def load_texts() -> dict:
    texts = {}
    csv = ROOT / "data/processed/natural-niche/all_queries_with_overlap.csv"
    if csv.exists():
        t = pd.read_csv(csv, index_col=0, dtype=str, usecols=lambda c: c in ("Unnamed: 0", "dataset", "query_text"))
        texts.update({f"{d}:{q}": s for q, d, s in zip(t.index, t["dataset"], t["query_text"]) if isinstance(s, str)})
    for base in [ROOT / "data/raw", ROOT / "data/downloads/queries_only"]:
        for path in base.glob("*/queries.jsonl"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    texts.setdefault(f"{path.parent.name}:{r['_id']}", r["text"])
    return texts


def load(scope: dict, texts: dict) -> pd.DataFrame:
    pub = pd.concat([pd.read_csv(ROOT / d / "per_query_heldout.csv", index_col=0) for d in scope["pub"]])
    pub.index = pub.index.astype(str)
    single = []
    for ds in pub["dataset"].unique():
        src = next(ROOT / d for d in scope["cache"] if (ROOT / d / f"{ds}.csv").exists())
        c = pd.read_csv(src / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        c.index = ds + ":" + c.index.astype(str)
        single.append(c[["ndcg_bm25", "ndcg_dense"]])
    df = pub.join(pd.concat(single), how="inner")
    df["text"] = pd.Series(texts).reindex(df.index)
    return nich.add_groups(df)


def macro(df: pd.DataFrame, col: str, groups: list, values: pd.Series) -> float:
    return float(np.mean([values[df[col] == g].mean() for g in groups]))


def boot_macro_diff(df: pd.DataFrame, col: str, groups: list, d: pd.Series, rng) -> list:
    """95% CI of the macro mean of d, resampling queries within each group."""
    means = np.zeros(N_BOOT)
    for g in groups:
        x = d[df[col] == g].to_numpy()
        idx = rng.integers(0, len(x), (N_BOOT, len(x)))
        means += x[idx].mean(axis=1)
    means /= len(groups)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def analyse(df: pd.DataFrame, min_n: int) -> dict:
    rng = np.random.default_rng(SEED)
    out = {}
    for col in GROUPINGS:
        sizes = df[col].dropna().value_counts()
        groups = sorted(sizes[sizes >= min_n].index.tolist(), key=str)
        if len(groups) < 2:
            continue
        sub = df[df[col].isin(groups)]
        res = {"n": int(len(sub)), "groups": {str(g): int(sizes[g]) for g in groups},
               "micro": {s: float(sub[c].mean()) for s, c in SYSTEMS.items()},
               "macro": {s: macro(sub, col, groups, sub[c]) for s, c in SYSTEMS.items()},
               "per_group": {str(g): {s: float(sub.loc[sub[col] == g, c].mean()) for s, c in SYSTEMS.items()} for g in groups}}
        res["diffs"] = {}
        for s in ("bm25", "dense", "router"):
            d = sub[SYSTEMS[s]] - sub[SYSTEMS["hybrid"]]
            res["diffs"][f"{s}_minus_hybrid"] = {"micro": float(d.mean()), "macro": macro(sub, col, groups, d),
                                                  "macro_ci": boot_macro_diff(sub, col, groups, d, rng)}
        out[col] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output_dir", type=Path, default=ROOT / "results/macro_ndcg_posthoc")
    ap.add_argument("--min_n", type=int, default=30)
    args = ap.parse_args()
    texts = load_texts()
    res, L = {}, ["# Category-balanced (macro) nDCG@10 over the E2 groupings — POST HOC", "",
                  "Not pre-registered. Built by `scripts/posthoc_macro_ndcg.py` from saved held-out predictions; no "
                  "retrieval was re-run. **Micro** = every query weighs the same (the usual average over the queries "
                  "the grouping covers). **Macro** = every group weighs the same (mean of the group means). Groupings "
                  f"and their definitions are E2's (`scripts/analyze_niche_existence.py`); groups with fewer than "
                  f"{args.min_n} queries are left out of that grouping. Hybrid = per-fold α* (Experiment 2d). Macro "
                  "differences: paired bootstrap resampling queries within each group, 10,000 resamples, seed 42. "
                  "The oracle is an upper bound, not a policy.", ""]
    for scope, cfg in SCOPES.items():
        df = load(cfg, texts)
        r = analyse(df, args.min_n)
        res[scope] = {"n_queries": int(len(df)), "datasets": sorted(df["dataset"].unique().tolist()),
                      "queries_without_text": int(df["text"].isna().sum()), "groupings": r}
        L += [f"## {scope.capitalize()} ({len(df):,} queries, {df['dataset'].nunique()} datasets"
              + (f"; {int(df['text'].isna().sum())} BRIGHT queries have no text and count only by dataset" if df["text"].isna().any() else "")
              + ")", "", "| Grouping | Groups (n) | Hybrid micro / macro | Router micro / macro | Oracle micro / macro | "
              "Router − hybrid, macro [95% CI] | BM25 − hybrid, macro | Dense − hybrid, macro |",
              "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
        for col, v in r.items():
            ci = lambda k: f"{v['diffs'][k]['macro']:+.4f} [{v['diffs'][k]['macro_ci'][0]:+.4f}, {v['diffs'][k]['macro_ci'][1]:+.4f}]"  # noqa: E731
            groups = ", ".join(f"{g} ({n})" for g, n in v["groups"].items())
            L.append(f"| {GROUPINGS[col]} | {groups} | {v['micro']['hybrid']:.4f} / {v['macro']['hybrid']:.4f} | "
                     f"{v['micro']['router']:.4f} / {v['macro']['router']:.4f} | {v['micro']['oracle']:.4f} / {v['macro']['oracle']:.4f} | "
                     f"{ci('router_minus_hybrid')} | {ci('bm25_minus_hybrid')} | {ci('dense_minus_hybrid')} |")
        L.append("")
    L += ["Per-group means of every system: `summary.json`."]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"posthoc": True, "min_n": args.min_n, **res}, indent=2), encoding="utf-8")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
