"""Does a routing niche exist, and does the Experiment 2d router win on it?

A niche (usable by a router) = a group of queries identifiable BEFORE retrieval (dataset / query form /
length / identifiers) where BM25 or dense beats hybrid on average. Per-query hindsight headroom
(oracle - hybrid) is reported separately: it exists, but it is only a niche if it can be found.

Inputs: published Experiment 2d held-out predictions (13 datasets) + Experiment 2 caches (per-query
BM25 / dense nDCG@10). Hybrid = Experiment 2d's per-fold tuned alpha. No retrieval is re-run.

Writes data/processed/natural-niche/niche_existence.md

Usage:
  python scripts/analyze_niche_existence.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402

OUT = ROOT / "data/processed/natural-niche/niche_existence.md"
N_BOOT, SEED = 5000, 42
CACHES = [ROOT / "results/niche/experiment2/cache", ROOT / "results/experiment2/cache"]
FAMILY = {"wands": "short keyword search", "cosqa": "short keyword search",
          "cqadupstack-android": "forum titles", "cqadupstack-webmasters": "forum titles",
          "cqadupstack-unix": "forum titles", "quora": "forum titles",
          "arguana": "long paragraphs", "bright-biology": "long paragraphs",
          "bright-economics": "long paragraphs", "bright-psychology": "long paragraphs",
          "scifact": "scientific claims / topics", "nfcorpus": "scientific claims / topics",
          "fiqa": "finance QA"}
QWORDS = {"what", "why", "how", "when", "where", "which", "who", "is", "are", "can", "could", "does", "do",
          "did", "should", "would", "will", "has", "have", "any"}
IDENT = re.compile(r"(\b(?=\w*\d)(?=\w*[a-zA-Z])\w+\b)|(\b[a-z]+[A-Z]\w*\b)|(\b\w+_\w+\b)|(\b\w+\.\w+\b)"
                   r"|((^|\s)--?\w+)|(\b[A-Z]{2,}\b)|(\"[^\"]+\")")


def load():
    pub = pd.concat([pd.read_csv(ROOT / "results/niche/experiment2d/per_query_heldout.csv", index_col=0),
                     pd.read_csv(ROOT / "results/experiment2d/per_query_heldout.csv", index_col=0)])
    pub.index = pub.index.astype(str)
    single = []
    for ds in pub["dataset"].unique():
        for d in CACHES:
            if (d / f"{ds}.csv").exists():
                c = pd.read_csv(d / f"{ds}.csv", index_col=0, dtype={"query_id": str})
                c.index = ds + ":" + c.index.astype(str)
                single.append(c[["ndcg_bm25", "ndcg_dense"]])
                break
    df = pub.join(pd.concat(single), how="inner")
    texts = {}
    for ds in df["dataset"].unique():
        path = ROOT / ("data/processed/quora-sub" if ds == "quora" else f"data/raw/{ds}") / "queries.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    texts[f"{ds}:{r['_id']}"] = r["text"]
    df["text"] = pd.Series(texts).reindex(df.index)
    return df


def add_groups(df):
    t = df["text"].fillna("")
    words = t.str.split().str.len()
    first = t.str.strip().str.split().str[0].str.lower().fillna("")
    df["g_family"] = df["dataset"].map(FAMILY)
    df["g_length"] = pd.cut(words, [0, 3, 7, 15, 50, 10_000], labels=["1-3 words", "4-7", "8-15", "16-50", ">50"])
    df["g_form"] = np.where(first.isin(QWORDS) | t.str.contains(r"\?"), "question", "statement/keywords")
    df["g_ident"] = np.where(t.apply(lambda s: bool(IDENT.search(s))), "has identifier-like token", "no identifier")
    missing = df["text"].isna()
    for g in ("g_length", "g_form", "g_ident"):
        df[g] = df[g].astype(object)
        df.loc[missing, g] = None  # BRIGHT texts were deleted with its raw data
    return df


def test(a, b):
    t = paired_bootstrap_test(np.asarray(a, float), np.asarray(b, float), N_BOOT, SEED)
    return t["delta_mean"], t["ci_95_low"], t["ci_95_high"], t["p_value"]


def fmt(t, star=True):
    d, lo, hi, _ = t
    mark = " ✅" if star and lo > 0 else (" ❌" if star and hi < 0 else "")
    return f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{mark}"


def holm(p):
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def group_table(df, col, label):
    rows = []
    for g, d in df.dropna(subset=[col]).groupby(col, observed=True):
        if len(d) < 20:
            continue
        h = d["hybrid_ndcg_cut_10"]
        rows.append({
            "group": g, "n": len(d),
            "bm25": test(d["ndcg_bm25"], h), "dense": test(d["ndcg_dense"], h),
            "headroom": (d["oracle_ndcg_cut_10"] - h).mean(),
            "share_gain10": ((d[["ndcg_bm25", "ndcg_dense"]].max(axis=1) - h) >= 0.10).mean(),
            "router": test(d["cost_aware_ndcg_cut_10"], h), "ungated": test(d["no_fallback_ndcg_cut_10"], h),
            "away": (d["route_cost_aware"] != "hybrid").mean(),
        })
    if not rows:
        return [], []
    p_niche = holm(np.array([min(r["bm25"][3], r["dense"][3]) * 2 for r in rows]).clip(max=1))
    p_router = holm(np.array([r["router"][3] for r in rows]))
    lines = [f"### By {label}", "",
             "| Group | n | BM25 − hybrid | Dense − hybrid | Niche? | Oracle headroom | Queries with a single "
             "retriever ≥ 0.10 above hybrid | Router − hybrid | Ungated router − hybrid | Routed away |",
             "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for r, pn, pr in zip(rows, p_niche, p_router):
        niche = ("BM25" if r["bm25"][1] > 0 else "dense" if r["dense"][1] > 0 else "no")
        lines.append(f"| {r['group']} | {r['n']} | {fmt(r['bm25'])} | {fmt(r['dense'])} | "
                     f"{niche}{' (Holm p=%.3f)' % pn if niche != 'no' else ''} | +{r['headroom']:.4f} | "
                     f"{r['share_gain10']:.1%} | {fmt(r['router'])}{' (Holm p=%.3f)' % pr if r['router'][1] > 0 else ''} | "
                     f"{fmt(r['ungated'])} | {r['away']:.1%} |")
    return lines + [""], rows


def main():
    df = add_groups(load())
    h = df["hybrid_ndcg_cut_10"]
    gain_best = df[["ndcg_bm25", "ndcg_dense"]].max(axis=1) - h
    md = ["# Does the routing niche exist, and does the router win on it?", "",
          f"{len(df)} held-out queries, {df['dataset'].nunique()} datasets (published Experiment 2d predictions; "
          "hybrid = per-fold tuned α). ✅ = 95% CI above 0, ❌ = below 0. `Niche?` = BM25 or dense beats hybrid "
          "on average in that group (CI > 0); Holm-adjusted across the groups in each table.", "",
          "## 1. Hindsight headroom (per query, not identifiable in advance)", "",
          f"- Oracle − hybrid: **+{(df['oracle_ndcg_cut_10'] - h).mean():.4f}** nDCG@10 pooled.",
          f"- Queries where BM25 or dense beats hybrid at all: **{(gain_best > 0).mean():.1%}**; by ≥ 0.10: "
          f"**{(gain_best >= 0.10).mean():.1%}**; by ≥ 0.20: **{(gain_best >= 0.20).mean():.1%}**.",
          f"- Queries where hybrid is worse than *both* single retrievers: "
          f"{((df['ndcg_bm25'] > h) & (df['ndcg_dense'] > h)).mean():.1%}.", ""]

    md.append("## 2. Is there an identifiable niche?")
    md.append("")
    all_rows = {}
    for col, label in [("dataset", "dataset"), ("g_family", "query family"), ("g_length", "query length"),
                       ("g_form", "query form"), ("g_ident", "identifier-like tokens (SKUs, versions, camelCase, "
                                                           "flags, ACRONYMS, quotes)")]:
        lines, rows = group_table(df, col, label)
        md += lines
        all_rows[col] = rows

    md += ["## 3. When the router leaves hybrid, is it right?", ""]
    for sys_, name in [("cost_aware", "Cost-aware GBM (published)"), ("no_fallback", "Same GBM without the fallback gate")]:
        away = df[df[f"route_{sys_}"] != "hybrid"]
        g = away[f"{sys_}_ndcg_cut_10"] - away["hybrid_ndcg_cut_10"]
        md.append(f"- **{name}**: left hybrid on {len(away)} queries ({len(away) / len(df):.1%}); gain vs hybrid on "
                  f"those = {fmt(test(away[f'{sys_}_ndcg_cut_10'], away['hybrid_ndcg_cut_10']))}; better / tied / "
                  f"worse = {(g > 0).mean():.0%} / {(g == 0).mean():.0%} / {(g < 0).mean():.0%}.")
    md.append("")

    niche_groups = [(c, r["group"]) for c, rows in all_rows.items() for r in rows
                    if r["bm25"][1] > 0 or r["dense"][1] > 0]
    router_wins = [(c, r["group"]) for c, rows in all_rows.items() for r in rows if r["router"][1] > 0]
    md += ["## 4. Verdict", "",
           f"- Groups where a single retriever significantly beats hybrid (uncorrected CI): "
           f"{', '.join(f'{g} ({c})' for c, g in niche_groups) or 'none'}.",
           f"- Groups where the router significantly beats hybrid (uncorrected CI): "
           f"{', '.join(f'{g} ({c})' for c, g in router_wins) or 'none'}.", ""]
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
